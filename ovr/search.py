"""Validation-only search for consensus refinement followed by dual-view fusion.

Native model fitting is independent of these nine operator parameters. Each
candidate retains fit-local OOF memories; no native model is refitted here.
"""

import copy
import itertools
import math
import time
from pathlib import Path

import numpy as np

from . import operators as op
from .artifacts import arrays, atomic_json, atomic_npz
from .evaluation import score

CONSENSUS_KEYS = ("huber_clip", "k", "alpha", "class_power", "pseudo")
FUSION_KEYS = ("beta_plus", "evidence_clip", "increment_clip", "tau")


def candidates(grid, keys):
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def search_budget(cfg, ntrain, nval, nclass):
    if not cfg["search"]["enabled"]:
        return dict(consensus_candidates=0, fusion_candidates=0, prediction_archive_mib=0.)
    search = cfg["search"]
    consensus_count = math.prod(len(search["consensus_refinement_grid"][k]) for k in CONSENSUS_KEYS)
    fusion_count = math.prod(len(search["dual_view_evidence_fusion_grid"][k]) for k in FUSION_KEYS) if cfg["dual_view_evidence_fusion"]["enabled"] else 0
    size = (consensus_count + fusion_count) * (ntrain + nval) * nclass * 8 / 1024 ** 2
    if consensus_count + fusion_count > search["max_candidates"]:
        raise ValueError("Search exceeds max_candidates; explicitly revise the grid or budget")
    if size > search["max_archive_mib"]:
        raise ValueError("Search prediction archive exceeds max_archive_mib")
    return dict(consensus_candidates=consensus_count, fusion_candidates=fusion_count, prediction_archive_mib=size)


def rank_key(metrics, uid, metric):
    return (-metrics[metric], -metrics["macro_f1"], -metrics["macro_auroc"], metrics["nll"], uid)


def _load_scopes(output, train_rows, train_y, val_rows, outer_folds):
    scopes = []
    seen = []
    for f in range(outer_folds):
        name = f"outer_{f}"
        training = arrays(output / "memory" / name / "training.npz")
        query = arrays(output / "native" / name / "raw_heldout.npz")
        positions = np.searchsorted(train_rows, query["row_index"])
        np.testing.assert_array_equal(train_rows[positions], query["row_index"])
        fit_positions = np.searchsorted(train_rows, training["row_index"])
        np.testing.assert_array_equal(train_rows[fit_positions], training["row_index"])
        np.testing.assert_array_equal(train_y[fit_positions], training["y"])
        if np.intersect1d(query["row_index"], training["row_index"]).size:
            raise ValueError("Outer query overlaps its inner-OOF memory")
        seen.extend(query["row_index"].tolist())
        scopes.append((name, training, query, positions))
    np.testing.assert_array_equal(np.sort(seen), train_rows)
    training = arrays(output / "memory" / "full_train" / "training.npz")
    np.testing.assert_array_equal(training["row_index"], train_rows)
    np.testing.assert_array_equal(training["y"], train_y)
    query = arrays(output / "native" / "full_train" / "raw_val.npz")
    np.testing.assert_array_equal(query["row_index"], val_rows)
    scopes.append(("full_train", training, query, None))
    return scopes


def _assemble_views(scopes, ntrain, nclass):
    specialized = np.empty((ntrain, nclass)); complete = np.empty_like(specialized)
    for _, _, query, positions in scopes[:-1]:
        specialized[positions] = query["specialized_probabilities"]; complete[positions] = query["complete_probabilities"]
    return dict(train_specialized=specialized, train_complete=complete,
                val_specialized=scopes[-1][2]["specialized_probabilities"], val_complete=scopes[-1][2]["complete_probabilities"])


def select_parameters(output, train_rows, train_y, val_rows, val_y, cfg, started, progress=None):
    """No Test arrays or Test paths are accepted by this function."""
    output = Path(output); search = cfg["search"]; nclass = len(np.unique(train_y))
    budget = search_budget(cfg, len(train_y), len(val_y), nclass)
    if not search["enabled"]:
        raise ValueError("Search is disabled")
    scopes = _load_scopes(output, train_rows, train_y, val_rows, cfg["split"]["outer_folds"])
    folder = output / "search"; folder.mkdir(exist_ok=False)
    consensus_grid = candidates(search["consensus_refinement_grid"], CONSENSUS_KEYS)
    maxk = max(search["consensus_refinement_grid"]["k"])
    geometry = {}
    for name, training, query, _ in scopes:
        state = op.build_residual_memory(training, consensus_grid[0]["huber_clip"])
        geometry[name] = op.neighbor_geometry(query, state, min(maxk, len(training["y"])))
    states = {}; previous_clip = None
    train_prob = np.empty((len(consensus_grid), len(train_y), nclass), dtype=np.float64)
    val_prob = np.empty((len(consensus_grid), len(val_y), nclass), dtype=np.float64)
    consensus_records = []

    def check_time():
        if time.perf_counter() - started > cfg["runtime"]["max_seconds"]:
            raise TimeoutError("Search reached the run's time budget; native fits remain saved")

    for index, params in enumerate(consensus_grid):
        check_time()
        if params["huber_clip"] != previous_clip:
            states = {name: op.build_residual_memory(t, params["huber_clip"]) for name, t, _, _ in scopes}
            previous_clip = params["huber_clip"]
        for name, _, query, positions in scopes:
            p, _ = op.refine_consensus(query, states[name], params, geom=geometry[name])
            if positions is None:
                val_prob[index] = p
            else:
                train_prob[index, positions] = p
        consensus_records.append(dict(uid=f"consensus_refined{index:06d}", parameters=params, validation_metrics=score(val_y, val_prob[index])))
    best_consensus = min(range(len(consensus_records)), key=lambda i: rank_key(consensus_records[i]["validation_metrics"], consensus_records[i]["uid"], search["metric"]))
    atomic_npz(folder / "consensus_refinement_development.npz", train_oof_p=train_prob, val_p=val_prob,
               train_oof_row_index=train_rows, train_oof_y=train_y, val_row_index=val_rows, val_y=val_y,
               parameter_names=np.asarray(CONSENSUS_KEYS), parameters=np.asarray([[p[k] for k in CONSENSUS_KEYS] for p in consensus_grid]))
    selected_train = train_prob[best_consensus].copy(); selected_val = val_prob[best_consensus].copy()
    del train_prob, val_prob
    selected_consensus = consensus_grid[best_consensus]
    selected_fusion = copy.deepcopy(cfg["dual_view_evidence_fusion"])
    fusion_records = []; best_fusion = None
    if progress:
        progress(f"Consensus refinement search complete: {len(consensus_grid)} candidates selected on validation")
    if cfg["dual_view_evidence_fusion"]["enabled"]:
        specialized = _assemble_views(scopes, len(train_y), nclass)
        fusion_grid = candidates(search["dual_view_evidence_fusion_grid"], FUSION_KEYS)
        train_prob = np.empty((len(fusion_grid), len(train_y), nclass), dtype=np.float64)
        val_prob = np.empty((len(fusion_grid), len(val_y), nclass), dtype=np.float64)
        for index, params in enumerate(fusion_grid):
            check_time()
            train_prob[index], _ = op.fuse_dual_view_evidence(selected_train, specialized["train_specialized"], specialized["train_complete"], params)
            val_prob[index], _ = op.fuse_dual_view_evidence(selected_val, specialized["val_specialized"], specialized["val_complete"], params)
            fusion_records.append(dict(uid=f"dual_view_evidence_fusion{index:06d}", parameters=params, validation_metrics=score(val_y, val_prob[index])))
        best_fusion = min(range(len(fusion_records)), key=lambda i: rank_key(fusion_records[i]["validation_metrics"], fusion_records[i]["uid"], search["metric"]))
        selected_fusion.update(fusion_grid[best_fusion])
        atomic_npz(folder / "dual_view_evidence_fusion_development.npz", train_oof_p=train_prob, val_p=val_prob,
                   train_oof_row_index=train_rows, train_oof_y=train_y, val_row_index=val_rows, val_y=val_y,
                   parameter_names=np.asarray(FUSION_KEYS), parameters=np.asarray([[p[k] for k in FUSION_KEYS] for p in fusion_grid]))
        if progress:
            progress(f"Dual-view evidence fusion search complete: {len(fusion_grid)} candidates selected on validation")
    selected = dict(consensus_refinement=selected_consensus, dual_view_evidence_fusion=selected_fusion)
    summary = dict(status="selected_on_validation", metric=search["metric"],
                   tie_breakers=["macro_f1_desc", "macro_auroc_desc", "nll_asc", "uid_asc"],
                   strategy="consensus_refinement_then_dual_view_evidence_fusion", budget=budget, consensus_candidates=consensus_records, fusion_candidates=fusion_records,
                   selected_consensus_index=best_consensus, selected_fusion_index=best_fusion, selected_parameters=selected,
                   test_used_for_selection=False,
                   oof_interpretation="fit-local nested memories; hyperparameters selected on shared validation, not nested HPO")
    atomic_json(folder / "selection.json", summary)
    return selected, summary
