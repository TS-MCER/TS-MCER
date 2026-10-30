"""Read the method and search configuration; accept only declared settings."""

import copy
import json
import math
import re
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.json"


def _merge(base, patch, prefix=""):
    if not isinstance(patch, dict):
        raise ValueError(f"{prefix or 'config'} must be a JSON object")
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value


def load_config(path=None):
    cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None:
        path = Path(path).resolve()
        patch = json.loads(path.read_text(encoding="utf-8-sig"))
        _merge(cfg, patch)
        checkpoint = cfg["model"]["checkpoint"]
        if checkpoint and not Path(checkpoint).is_absolute():
            cfg["model"]["checkpoint"] = str((path.parent / checkpoint).resolve())
    validate(cfg)
    return cfg


def validate(cfg):
    for section, keys in {
        "data": ("features", "exclude_columns", "binary_columns"),
        "dual_view_evidence_fusion": ("specialized_columns",),
    }.items():
        for key in keys:
            value = cfg[section][key]
            if value is None and key in ("features", "binary_columns"):
                continue
            if (not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value)
                    or len(value) != len(set(value))):
                raise ValueError(f"{section}.{key} must contain unique column names")
    d = cfg["data"]
    for key in ("id_column", "group_column"):
        if d[key] is not None and (not isinstance(d[key], str) or not d[key]):
            raise ValueError(f"data.{key} must be a column name or null")
    if not isinstance(d["missing_suffix"], str) or not d["missing_suffix"]:
        raise ValueError("data.missing_suffix must be nonempty")
    if not isinstance(d["missing_values"], list) or any(not isinstance(v, str) for v in d["missing_values"]):
        raise ValueError("data.missing_values must contain strings")
    for section, key, minimum in [
        ("split", "seed", 0), ("split", "outer_folds", 2), ("split", "inner_folds", 2),
        ("model", "n_estimators", 1), ("model", "preprocessing_jobs", 1),
        ("model", "cpu_threads", 1), ("consensus_refinement", "k", 1),
    ]:
        v = cfg[section][key]
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    for section, keys in {
        "split": ("train_fraction", "val_fraction"),
        "consensus_refinement": ("huber_clip", "alpha", "class_power", "pseudo"),
        "dual_view_evidence_fusion": ("beta_plus", "evidence_clip", "increment_clip", "tau"),
        "runtime": ("max_seconds", "min_free_gpu_gib"),
    }.items():
        for key in keys:
            v = cfg[section][key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{section}.{key} must be a finite nonnegative number")
    s = cfg["split"]
    if not (0 < s["train_fraction"] < 1 and 0 < s["val_fraction"] < 1
            and s["train_fraction"] + s["val_fraction"] < 1):
        raise ValueError("Train and validation fractions must leave a nonempty test split")
    if cfg["consensus_refinement"]["huber_clip"] <= 0 or cfg["runtime"]["max_seconds"] <= 0:
        raise ValueError("huber_clip and max_seconds must be positive")
    for section, key in [("dual_view_evidence_fusion", "enabled"), ("runtime", "verify_reload")]:
        if not isinstance(cfg[section][key], bool):
            raise ValueError(f"{section}.{key} must be boolean")
    if cfg["dual_view_evidence_fusion"]["enabled"] and not cfg["dual_view_evidence_fusion"]["specialized_columns"]:
        raise ValueError("dual_view_evidence_fusion.enabled requires dual_view_evidence_fusion.specialized_columns (a proper subset of feature columns)")
    if not isinstance(cfg["model"]["device"], str):
        raise ValueError("model.device must be a device string")
    checkpoint = cfg["model"]["checkpoint"]
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError("model.checkpoint must be a local path or null")
    checksum = cfg["model"]["checkpoint_sha256"]
    if not isinstance(checksum, str) or not re.fullmatch("[0-9a-f]{64}", checksum):
        raise ValueError("model.checkpoint_sha256 must be the expected weight SHA256")
    search = cfg["search"]
    if not isinstance(search["enabled"], bool):
        raise ValueError("search.enabled must be boolean")
    if search["metric"] not in ("macro_f1", "accuracy", "macro_auroc", "composite"):
        raise ValueError("Unsupported validation selection metric")
    for key in ("max_candidates", "max_archive_mib"):
        if isinstance(search[key], bool) or not isinstance(search[key], int) or search[key] <= 0:
            raise ValueError(f"search.{key} must be a positive integer")
    for section in ("consensus_refinement_grid", "dual_view_evidence_fusion_grid"):
        for key, values in search[section].items():
            if not isinstance(values, list) or not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or v < 0 for v in values) or len(set(values)) != len(values):
                raise ValueError(f"search.{section}.{key} needs unique finite nonnegative numeric values")
            if key == "k" and any(not isinstance(v, int) or v < 1 for v in values):
                raise ValueError("Search k values must be positive integers")
            if key == "huber_clip" and min(values) <= 0:
                raise ValueError("Search Huber clips must be positive")


def public_config(cfg):
    """Do not embed a machine's checkpoint path into the summary."""
    result = copy.deepcopy(cfg)
    result["model"]["checkpoint"] = None
    return result
