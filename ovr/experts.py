"""Local TabPFN base_probabilities and all-class OvR experts with saved fit scopes."""

import gc
import importlib.metadata
import os
import time
from pathlib import Path

import joblib
import numpy as np

from .artifacts import atomic_json, atomic_npz, digest
from .data import FoldPreprocessor
from .operators import normalize


def coverage(rows, x, n, width):
    if n == 0:
        return np.asarray([], dtype=int)
    if not 0 <= n <= len(rows):
        raise ValueError("Invalid coverage budget")
    x = x.astype(float).copy(); x[:, :width] = np.clip(x[:, :width], -5, 5); x /= np.sqrt(float(width))
    d = ((x - x.mean(0)) ** 2).sum(1)
    first = np.lexsort((rows, d))[0]
    selected = [int(first)]; used = np.zeros(len(rows), bool); used[first] = True
    near = ((x - x[first]) ** 2).sum(1)
    for _ in range(n - 1):
        options = np.flatnonzero(~used); j = options[np.lexsort((rows[options], -near[options]))[0]]
        selected.append(int(j)); used[j] = True
        near = np.minimum(near, ((x - x[j]) ** 2).sum(1))
    return rows[selected]


def quota(counts, total):
    counts = np.asarray(counts, int)
    if not np.count_nonzero(counts) <= total <= counts.sum():
        raise ValueError("Invalid proportional sampling budget")
    ideal = total * counts / counts.sum()
    q = np.minimum(counts, np.maximum((counts > 0).astype(int), np.floor(ideal).astype(int)))
    while q.sum() > total:
        choices = np.flatnonzero(q > 1); j = choices[np.lexsort((choices, -(q[choices] - ideal[choices])))[0]]; q[j] -= 1
    while q.sum() < total:
        choices = np.flatnonzero(q < counts); j = choices[np.lexsort((choices, -(ideal[choices] - q[choices])))[0]]; q[j] += 1
    return q


def contexts(y, fit, x, width, nclass, seed, scope_id):
    """x has global row indexing, but only fit rows are ever used for sampling."""
    for c in range(nclass):
        positive = fit[y[fit] == c]; negative = fit[y[fit] != c]
        counts = np.bincount(y[negative], minlength=nclass)
        budget = min(len(negative), max(int(np.ceil(.5 * len(negative))), np.count_nonzero(counts)))
        quotas = quota(counts, budget)
        cov_total = min(budget, max(np.count_nonzero(counts), int(np.ceil(.2 * budget))))
        cov_quotas = quota(counts, cov_total)
        # Preserve the original rule; never silently alter a problematic quota.
        if np.any(cov_quotas > quotas):
            raise ValueError("Coverage quota exceeds the sampled budget for a class")
        cov = []
        for d in range(nclass):
            dr = negative[y[negative] == d]
            cov.append(coverage(dr, x[dr], int(cov_quotas[d]), width))
        for j, offset in enumerate((0, 100, 200)):
            rng = np.random.default_rng(seed + 10000 * (scope_id + 1) + offset)
            selected = [positive]
            for d in range(nclass):
                remaining = negative[(y[negative] == d) & ~np.isin(negative, cov[d])]
                selected.extend([cov[d], rng.choice(remaining, int(quotas[d] - len(cov[d])), replace=False)])
            rows = np.sort(np.concatenate(selected))
            if len(np.unique(rows)) != len(positive) + budget or not np.isin(positive, rows).all():
                raise ValueError("Invalid expert context")
            yield c, j, rows


class TabPFNBackend:
    """No hosted client, automatic weight download, or fallback model."""

    def __init__(self, cfg):
        checkpoint = cfg["model"]["checkpoint"] or os.environ.get("OVR_CHECKPOINT")
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("Set OVR_CHECKPOINT or model.checkpoint to an existing local TabPFN-3.5 weight file")
        self.checkpoint = Path(checkpoint).resolve()
        checksum = digest(self.checkpoint)
        if checksum != cfg["model"]["checkpoint_sha256"]:
            raise ValueError("Local checkpoint hash does not match the frozen TabPFN-3.5 recipe")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            version = importlib.metadata.version("tabpfn")
        except importlib.metadata.PackageNotFoundError:
            raise ValueError("Install the dependencies from requirements.txt before fitting") from None
        if version != "9.1.0":
            raise ValueError("This recipe is validated with tabpfn==9.1.0; do not silently change the model implementation")
        import torch
        from tabpfn import TabPFNClassifier
        self.torch = torch; self.cls = TabPFNClassifier; self.cfg = cfg
        device = cfg["model"]["device"]
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        torch.set_num_threads(cfg["model"]["cpu_threads"])
        self.info = dict(backend="TabPFN-3.5", package_version=version, device=self.device,
                         checkpoint_sha256=checksum, torch_version=torch.__version__)

    def fit(self, x, y, seed):
        if self.device.startswith("cuda"):
            free, _ = self.torch.cuda.mem_get_info(self.torch.device(self.device))
            if free < self.cfg["runtime"]["min_free_gpu_gib"] * 1024 ** 3:
                raise RuntimeError("Insufficient free GPU memory")
        model = self.cls(n_estimators=self.cfg["model"]["n_estimators"], random_state=seed,
                         model_path=str(self.checkpoint), device=self.device,
                         fit_mode="fit_preprocessors", memory_saving_mode="auto", inference_precision="auto",
                         softmax_temperature=1., balance_probabilities=False, average_before_softmax=False,
                         n_preprocessing_jobs=self.cfg["model"]["preprocessing_jobs"], n_jobs=1,
                         eval_metric=None, tuning_config=None)
        model.fit(x, y)
        if model.n_train_samples_ != len(y) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise RuntimeError("Unexpected internal subsampling")
        return model

    def save(self, model, path):
        model.save_fit_state(path)

    def load(self, path):
        return self.cls.load_from_fit_state(path, device=self.device)

    def release(self):
        gc.collect()
        if self.device.startswith("cuda"):
            self.torch.cuda.empty_cache()


class NativeRunner:
    def __init__(self, data, split, cfg, output, backend, started):
        self.data = data; self.split = split; self.cfg = cfg; self.output = Path(output)
        self.backend = backend; self.started = started; self.models = {}; self.scopes = {}
        self.nclass = len(data.classes); self.max_reload_error = 0.

    def check_budget(self):
        if time.perf_counter() - self.started > self.cfg["runtime"]["max_seconds"]:
            raise TimeoutError("Runtime budget reached; completed local artifacts were preserved")

    def primitive(self, scope, fit, context, rows, kind, seed, pp, x, tag, target=None, allow_fit=True):
        self.check_budget()
        if not np.isin(context, fit).all() or not np.isin(fit, self.split["train"]).all() or np.intersect1d(fit, rows).size:
            raise ValueError("Illegal fit/query scope")
        folder = self.output / "native" / scope / kind; folder.mkdir(parents=True, exist_ok=True)
        modelpath = folder / "model.tabpfn_fit"
        expected = np.arange(self.nclass if target is None else 2)
        labels = self.data.y[context] if target is None else (self.data.y[context] == target).astype(int)
        if not np.array_equal(np.unique(labels), expected):
            raise ValueError("A native context is missing a class")
        key = scope + "/" + kind
        if key in self.models:
            model = self.backend.load(modelpath)
        else:
            if not allow_fit or modelpath.exists():
                raise RuntimeError("Test prediction must reuse a completed fit; incomplete states cannot be overwritten")
            atomic_npz(folder / "context.npz", fit_rows=fit, row_index=context, context_y=labels)
            model = self.backend.fit(pp.transform(x[context]), labels, seed)
            pending = folder / "pending.tabpfn_fit"
            self.backend.save(model, pending); os.replace(pending, modelpath)
            self.models[key] = dict(model=str(modelpath.relative_to(self.output).as_posix()),
                                    context=str((folder / "context.npz").relative_to(self.output).as_posix()),
                                    model_seed=seed, context_size=len(context), target_index=target)
        np.testing.assert_array_equal(model.classes_, expected)
        query = pp.transform(x[rows])
        p = normalize(model.predict_proba(query))
        if p.shape != (len(rows), len(expected)):
            raise ValueError("Wrong native probability dimensions")
        del model; self.backend.release()
        error = 0.
        if self.cfg["runtime"]["verify_reload"]:
            model = self.backend.load(modelpath)
            q = normalize(model.predict_proba(query))
            np.testing.assert_allclose(p, q, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(p.argmax(1), q.argmax(1))
            error = float(abs(p - q).max()); self.max_reload_error = max(self.max_reload_error, error)
            del model; self.backend.release()
        atomic_npz(folder / f"{tag}.npz", row_index=rows, p=p, reload_max_error=np.array(error))
        self.check_budget()
        return p

    def raw(self, scope, scope_id, fit, rows, tag, dual_view=False, allow_fit=True):
        self.check_budget()
        if self.data.groups is not None and np.intersect1d(self.data.groups[fit], self.data.groups[rows]).size:
            raise ValueError("Fit/query groups overlap")
        folder = self.output / "native" / scope; folder.mkdir(parents=True, exist_ok=True)
        if scope not in self.scopes:
            if not allow_fit:
                raise RuntimeError("Missing fitted scope")
            pp = FoldPreprocessor(self.data.names, self.data.binary).fit(self.data.x[fit])
            joblib.dump(pp, folder / "preprocessor.joblib")
            atomic_npz(folder / "rows.npz", fit_rows=fit)
            self.scopes[scope] = fit.copy()
        else:
            np.testing.assert_array_equal(fit, self.scopes[scope])
            pp = joblib.load(folder / "preprocessor.joblib")
        members = [self.primitive(scope, fit, fit, rows, f"base_multiclass_{seed}", seed, pp, self.data.x, tag,
                                  allow_fit=allow_fit) for seed in (42, 43, 44)]
        p = normalize(np.mean(members, axis=0)); opinions = np.empty((len(rows), self.nclass, 3))
        # Do not transform held-out data for context selection.
        fit_x = np.zeros((len(self.data.y), 2 * len(self.data.names)), dtype=np.float32)
        fit_x[fit] = pp.transform(self.data.x[fit])
        for c, j, context in contexts(self.data.y, fit, fit_x, len(self.data.names), self.nclass,
                                      self.cfg["split"]["seed"], scope_id):
            opinions[:, c, j] = self.primitive(scope, fit, context, rows, f"opinion_{c}_{j}", 42, pp,
                                               self.data.x, tag, target=c, allow_fit=allow_fit)[:, 1]
        z = dict(row_index=rows, base_probabilities=p, context_probabilities=opinions)
        if dual_view:
            names = self.cfg["dual_view_evidence_fusion"]["specialized_columns"]
            indices = [self.data.names.index(n) for n in names]; specialized_x = self.data.x[:, indices]
            if not (folder / "specialized_preprocessor.joblib").exists():
                binary = None if self.data.binary is None else [n for n in self.data.binary if n in names]
                specialized_pp = FoldPreprocessor(names, binary).fit(specialized_x[fit])
                joblib.dump(specialized_pp, folder / "specialized_preprocessor.joblib")
            else:
                specialized_pp = joblib.load(folder / "specialized_preprocessor.joblib")
            specialized, complete = np.empty_like(p), np.empty_like(p)
            for c in range(self.nclass):
                specialized[:, c] = self.primitive(scope, fit, fit, rows, f"specialized_view_{c}", 42, specialized_pp, specialized_x, tag,
                                             target=c, allow_fit=allow_fit)[:, 1]
                complete[:, c] = self.primitive(scope, fit, fit, rows, f"complete_view_{c}", 42, pp, self.data.x, tag,
                                             target=c, allow_fit=allow_fit)[:, 1]
            z.update(specialized_probabilities=specialized, complete_probabilities=complete)
        atomic_npz(folder / f"raw_{tag}.npz", **z)
        return z
