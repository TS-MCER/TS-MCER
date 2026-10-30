"""Multi-context expert consensus, local residual refinement, and optional evidence fusion."""

import copy
import importlib.metadata
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import __version__, operators
from .artifacts import arrays, atomic_json, atomic_npz, digest
from .config import public_config
from .data import describe
from .evaluation import paired, score
from .experts import NativeRunner, TabPFNBackend


def _predict(raw, state, cfg):
    consensus_refined, extra = operators.refine_consensus(raw, state, cfg["consensus_refinement"])
    final = consensus_refined.copy()
    extra.update(base_probabilities=raw["base_probabilities"], context_probabilities=raw["context_probabilities"],
                 evidence_increment=np.zeros_like(consensus_refined))
    if cfg["dual_view_evidence_fusion"]["enabled"]:
        final, increment = operators.fuse_dual_view_evidence(consensus_refined, raw["specialized_probabilities"],
                                      raw["complete_probabilities"], cfg["dual_view_evidence_fusion"])
        extra.update(specialized_probabilities=raw["specialized_probabilities"],
                     complete_probabilities=raw["complete_probabilities"], evidence_increment=increment)
    return dict(p=final, **extra)


def _save_memory(output, scope, training, cfg, directory="memory"):
    state = operators.build_residual_memory(training, cfg["consensus_refinement"]["huber_clip"])
    if directory == "memory":
        atomic_npz(output / directory / scope / "training.npz", **training)
    atomic_npz(output / directory / scope / "state.npz", **state)
    return state


def _run(data, split, cfg, output, backend, started, input_sha256, progress, code_hashes):
    description = describe(data, split, cfg)
    native = NativeRunner(data, split, cfg, output, backend, started)
    train = split["train"]; fold = split["outer_fold"][train]
    nclass = len(data.classes); outer = cfg["split"]["outer_folds"]; inner = cfg["split"]["inner_folds"]
    buffers = {}
    split_values = dict(**split, y=data.y)
    if data.groups is not None:
        split_values["group_code"] = data.groups
    atomic_npz(output / "splits.npz", **split_values)
    atomic_json(output / "schema.json", dict(label=data.label, class_values=data.classes,
                                              feature_columns=data.names, binary_columns=data.binary,
                                              row_index="zero-based position among CSV data rows"))
    for f in range(outer):
        fit, hold = train[fold != f], train[fold == f]
        scope = f"outer_{f}"
        local_fold = split["inner_fold"][f, fit]
        training = dict(row_index=fit, y=data.y[fit], fold=local_fold,
                        base_probabilities=np.full((len(fit), nclass), np.nan), context_probabilities=np.full((len(fit), nclass, 3), np.nan))
        for i in range(inner):
            inner_fit, inner_hold = fit[local_fold != i], fit[local_fold == i]
            raw = native.raw(f"{scope}/inner_{i}", 1 + outer + inner * f + i, inner_fit, inner_hold, "heldout")
            mask = local_fold == i
            training["base_probabilities"][mask] = raw["base_probabilities"]; training["context_probabilities"][mask] = raw["context_probabilities"]
        state = _save_memory(output, scope, training, cfg)
        raw = native.raw(scope, 1 + f, fit, hold, "heldout", dual_view=cfg["dual_view_evidence_fusion"]["enabled"])
        result = _predict(raw, state, cfg)
        for name, values in result.items():
            if name not in buffers:
                buffers[name] = np.full((len(train), *values.shape[1:]), np.nan)
            buffers[name][fold == f] = values
        atomic_npz(output / "predictions" / f"outer_{f}.npz", row_index=hold, y=data.y[hold],
                   fold=np.full(len(hold), f), **result)
        if progress:
            progress(f"Completed outer fold {f + 1}/{outer}; native fits {len(native.models)}")
    if not all(np.isfinite(v).all() for v in buffers.values()):
        raise RuntimeError("Incomplete outer OOF predictions")
    training = dict(row_index=train, y=data.y[train], fold=fold, base_probabilities=buffers["base_probabilities"], context_probabilities=buffers["context_probabilities"])
    state = _save_memory(output, "full_train", training, cfg)
    atomic_npz(output / "predictions" / "train_oof.npz", row_index=train, y=data.y[train], fold=fold, **buffers)
    val_raw = native.raw("full_train", 0, train, split["val"], "val", dual_view=cfg["dual_view_evidence_fusion"]["enabled"])
    effective = copy.deepcopy(cfg)
    search_result = None
    if cfg["search"]["enabled"]:
        from .search import select_parameters
        selected, selection = select_parameters(output, train, data.y[train], split["val"],
                                                 data.y[split["val"]], cfg, started, progress)
        effective["consensus_refinement"] = selected["consensus_refinement"]
        effective["dual_view_evidence_fusion"] = selected["dual_view_evidence_fusion"]
        search_result = dict(selection="search/selection.json", selected_parameters=selected,
                             budget=selection["budget"], test_used_for_selection=False)
        buffers = {}
        for f in range(outer):
            scope = f"outer_{f}"
            t = arrays(output / "memory" / scope / "training.npz")
            local_state = _save_memory(output, scope, t, effective, directory="selected_memory")
            raw = arrays(output / "native" / scope / "raw_heldout.npz")
            result = _predict(raw, local_state, effective)
            for name, values in result.items():
                if name not in buffers:
                    buffers[name] = np.full((len(train), *values.shape[1:]), np.nan)
                buffers[name][fold == f] = values
            hold = train[fold == f]
            atomic_npz(output / "predictions" / f"outer_{f}.npz", row_index=hold, y=data.y[hold],
                       fold=np.full(len(hold), f), **result)
        if not all(np.isfinite(v).all() for v in buffers.values()):
            raise RuntimeError("Incomplete selected OOF predictions")
        atomic_npz(output / "predictions" / "train_oof.npz", row_index=train, y=data.y[train], fold=fold, **buffers)
        state = _save_memory(output, "full_train", training, effective, directory="selected_memory")
    for part in ("val", "test"):
        rows = split[part]
        if part == "val":
            raw = val_raw
        else:
            atomic_json(output / "development_complete.json", dict(status="development_complete",
                         config=public_config(effective), search=search_result,
                         native_fit_count=len(native.models), input_sha256=input_sha256))
            raw = native.raw("full_train", 0, train, rows, "test", dual_view=cfg["dual_view_evidence_fusion"]["enabled"], allow_fit=False)
        result = _predict(raw, state, effective)
        atomic_npz(output / "predictions" / f"{part}.npz", row_index=rows, y=data.y[rows],
                   fold=np.full(len(rows), -1), **result)
    if len(native.models) != description["native_fit_count"]:
        raise RuntimeError("Unexpected number of native fits")
    metrics, comparisons, stage_metrics = {}, {}, {}
    for part in ("train_oof", "val", "test"):
        z = arrays(output / "predictions" / f"{part}.npz")
        rows = train if part == "train_oof" else split[part]
        np.testing.assert_array_equal(z["row_index"], rows)
        np.testing.assert_array_equal(z["y"], data.y[rows])
        metrics[part] = score(z["y"], z["p"])
        stage_metrics[part] = {stage: score(z["y"], z[stage]) for stage in ("base_probabilities", "expert_consensus", "consensus_refined")}
        comparisons[part] = {a + "_to_" + b: paired(z["y"], z[a], z[b])
                             for a, b in (("base_probabilities", "expert_consensus"), ("expert_consensus", "consensus_refined"), ("consensus_refined", "p"))}
    # One model manifest; no source machine paths or original identifiers.
    atomic_json(output / "native" / "models.json", native.models)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")}
    summary = dict(status="complete", package_version=__version__, config=public_config(cfg),
                   effective_config=public_config(effective), search=search_result,
                   input_sha256=input_sha256, input=description, backend=backend.info,
                   environment=dict(python=platform.python_version(), packages=versions),
                   code_sha256=code_hashes,
                   native_fit_count=len(native.models), metrics=metrics, stage_metrics=stage_metrics,
                   stage_comparisons=comparisons, elapsed_seconds=time.perf_counter() - started,
                   verification=dict(predictions_reloaded=True, strict_nested_oof=True,
                                     native_reload_checked=cfg["runtime"]["verify_reload"],
                                     native_reload_max_error=native.max_reload_error if cfg["runtime"]["verify_reload"] else None),
                   artifacts=dict(predictions="predictions/", splits="splits.npz", schema="schema.json",
                                  native_models="native/models.json", memory_training="memory/",
                                  memories="selected_memory/" if search_result else "memory/"))
    atomic_json(output / "result.json", summary)
    return summary


def run_pipeline(data, split, cfg, output, *, input_sha256=None, backend=None, progress=print):
    """backend injection is for tests; the CLI always uses real local TabPFN."""
    describe(data, split, cfg)  # Validate before creating anything.
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory to preserve previous artifacts")
    code_root = Path(__file__).resolve().parents[1]
    code_files = [code_root / "main.py", code_root / "config.json", *sorted((code_root / "ovr").glob("*.py"))]
    code_hashes = {p.relative_to(code_root).as_posix(): digest(p) for p in code_files}
    output.mkdir(parents=True)
    started = time.perf_counter()
    atomic_json(output / "run.json", dict(status="running", config=public_config(cfg), input_sha256=input_sha256))
    try:
        with runtime_environment(output), threadpool_limits(cfg["model"]["cpu_threads"]):
            if backend is None:
                backend = TabPFNBackend(cfg)
            result = _run(data, split, cfg, output, backend, started, input_sha256, progress, code_hashes)
    except BaseException as error:
        # Tracebacks can contain values or machine paths. Keep only the error type
        # here; the CLI reports a short diagnostic without dumping input rows.
        atomic_json(output / "run.json", dict(status="failed", error_type=type(error).__name__,
                                               config=public_config(cfg), input_sha256=input_sha256))
        raise
    atomic_json(output / "run.json", dict(status="complete", result="result.json"))
    return result


@contextmanager
def runtime_environment(output):
    """Set library caches before importing TabPFN/skrub, then restore the caller."""
    scratch = Path(output) / "scratch"; scratch.mkdir()
    values = {}
    for key in ("TEMP", "TMP", "JOBLIB_TEMP_FOLDER", "CUDA_CACHE_PATH", "SKB_DATA_DIRECTORY",
                "XDG_CACHE_HOME", "TORCH_HOME", "MPLCONFIGDIR"):
        folder = scratch / key.lower(); folder.mkdir(); values[key] = str(folder)
    for key in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TABPFN_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE"):
        values[key] = "1"
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
