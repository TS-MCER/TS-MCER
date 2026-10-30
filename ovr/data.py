"""CSV contract, record/group splits, and fit-local numeric preprocessing."""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split


# These are column names, never records. Unknown identifier columns still need
# explicit exclusion by the data owner.
ID_NAMES = {"id", "record_id", "sample_id", "subject_id", "person_id", "group_id",
            "name", "email", "phone"}


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    classes: list
    names: list
    binary: list
    groups: np.ndarray | None
    ignored: list
    checked_flags: list
    label: str


def _class_order(values):
    values = sorted(set(values))
    try:
        numeric = [Decimal(v) for v in values]
        if all(v.is_finite() for v in numeric):
            return [v for _, v in sorted(zip(numeric, values))]
    except InvalidOperation:
        pass
    return values


def read_csv(path, label, cfg):
    d = cfg["data"]
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or any(not h or h != h.strip() for h in header) or len(set(header)) != len(header):
            raise ValueError("CSV needs unique, nonempty headers without surrounding whitespace")
        rows = list(reader)
    if not rows or any(len(row) != len(header) for row in rows):
        raise ValueError("CSV is empty or contains rows with an inconsistent column count")
    if label not in header:
        raise ValueError("The specified label column is absent from the CSV")
    index = {name: i for i, name in enumerate(header)}
    missing = set(d["missing_values"])
    label_missing = {"", "?", "NA", "NaN", "nan", "null"}
    labels = [r[index[label]].strip() for r in rows]
    if any(v in label_missing for v in labels):
        raise ValueError("Missing target values are not allowed; no rows were dropped")
    classes = _class_order(labels)
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    codes = {value: i for i, value in enumerate(classes)}
    y = np.asarray([codes[v] for v in labels], dtype=np.int64)
    requested = list(d["exclude_columns"]) + [v for v in (d["id_column"], d["group_column"]) if v]
    if any(name not in index for name in requested):
        raise ValueError("A configured excluded/id/group column is absent from the CSV")
    if label in requested:
        raise ValueError("The label cannot also be an identifier, group, or excluded column")
    protected = {name for name in header if name.casefold() in ID_NAMES}
    excluded = set(requested) | protected | {label}
    suffix = d["missing_suffix"]
    flags = [name for name in header if name.endswith(suffix)]
    for name in flags:
        if name[:-len(suffix)] not in index:
            raise ValueError("A stored missing-indicator column has no corresponding value column")
    names = d["features"]
    if names is None:
        names = [h for h in header if h not in excluded and h not in flags]
    if not names or any(n not in index or n in excluded or n in flags for n in names):
        raise ValueError("Feature selection is empty, absent, or includes a target/id/missing-indicator column")
    x = np.empty((len(rows), len(names)), dtype=np.float64)
    checked = []
    for j, name in enumerate(names):
        values = [r[index[name]].strip() for r in rows]
        for i, value in enumerate(values):
            if value in missing:
                x[i, j] = np.nan
            else:
                try:
                    x[i, j] = float(value)
                except ValueError:
                    raise ValueError(f"Non-numeric feature at row {i + 2}, column {j + 1}; encode it explicitly") from None
                if not np.isfinite(x[i, j]):
                    raise ValueError(f"Nonfinite feature at row {i + 2}, column {j + 1}; declare missing tokens explicitly")
        flag = name + suffix
        if flag in index:
            raw = [r[index[flag]].strip() for r in rows]
            if any(v not in ("0", "1", "0.0", "1.0") for v in raw):
                raise ValueError("Stored missing indicators must be 0 or 1")
            if not np.array_equal(np.asarray(raw, float), np.isnan(x[:, j]).astype(float)):
                raise ValueError("Stored missing indicators disagree with value columns")
            checked.append(flag)
    binary = d["binary_columns"]
    if binary is not None and any(n not in names for n in binary):
        raise ValueError("Every binary column must be a selected feature")
    groups = None
    if d["group_column"]:
        raw = [r[index[d["group_column"]]].strip() for r in rows]
        if any(v in label_missing for v in raw):
            raise ValueError("Grouping identifiers must not be missing")
        mapping = {value: i for i, value in enumerate(dict.fromkeys(raw))}
        groups = np.asarray([mapping[v] for v in raw], dtype=np.int64)
        if any(len(np.unique(y[groups == g])) != 1 for g in np.unique(groups)):
            raise ValueError("A group has conflicting labels; resolve the prediction unit before splitting")
    if d["id_column"] and not d["group_column"]:
        raw = [r[index[d["id_column"]]].strip() for r in rows]
        if any(v in label_missing for v in raw) or len(set(raw)) != len(raw):
            raise ValueError("id_column must be complete and unique; use group_column for repeated subjects")
    return Dataset(x, y, classes, list(names), binary, groups,
                   [h for h in header if h not in names and h != label and h not in checked], checked, label)


class FoldPreprocessor:
    """NaN is internal missingness; never fit statistics on query rows."""

    def __init__(self, names, binary_columns=None):
        self.names = list(names)
        self.binary_columns = binary_columns

    def _check(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.names) or np.isinf(x).any():
            raise ValueError("Invalid feature matrix")
        return x

    def fit(self, x):
        x = self._check(x)
        if not len(x):
            raise ValueError("Empty fit scope")
        # Type inference, like all statistics, is fit-local. The default matches
        # binary 0/1 inputs; explicitly specify [] for numeric ordinal 0/1 fields.
        if self.binary_columns is None:
            self.binary = np.asarray([np.array_equal(np.unique(c[np.isfinite(c)]), [0., 1.]) for c in x.T])
        else:
            self.binary = np.asarray([name in self.binary_columns for name in self.names])
        self.fill = np.zeros(x.shape[1]); self.mean = np.zeros(x.shape[1]); self.std = np.ones(x.shape[1])
        for j, column in enumerate(x.T):
            observed = column[np.isfinite(column)]
            if self.binary[j]:
                if not np.isin(observed, [0, 1]).all():
                    raise ValueError("Declared binary column contains values outside 0/1")
            elif len(observed):
                self.fill[j] = np.median(observed)
                self.mean[j] = observed.mean()
                deviation = observed.std()
                self.std[j] = deviation if deviation > 1e-12 else 1.
        return self

    def transform(self, x):
        if not hasattr(self, "fill"):
            raise RuntimeError("Preprocessor is not fitted")
        x = self._check(x)
        if self.binary_columns is not None and not np.isin(x[:, self.binary][np.isfinite(x[:, self.binary])], [0, 1]).all():
            raise ValueError("Declared binary column contains values outside 0/1")
        missing = np.isnan(x)
        values = (np.where(missing, self.fill, x) - self.mean) / self.std
        result = np.concatenate([values, missing.astype(float)], axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessed values exceed finite float32 range")
        return result


def make_split(data, cfg):
    s = cfg["split"]; seed = s["seed"]; y = data.y; k = len(data.classes)
    if data.groups is None:
        units = np.arange(len(y)); unit_y = y
    else:
        units, first = np.unique(data.groups, return_index=True); unit_y = y[first]
    ntrain = int(np.floor(len(units) * s["train_fraction"]))
    nval = int(np.floor(len(units) * s["val_fraction"] + .5))
    if min(ntrain, nval, len(units) - ntrain - nval) < k:
        raise ValueError("Too few samples/groups to represent every class in all three splits")
    tr, rest = train_test_split(units, train_size=ntrain, stratify=unit_y, random_state=seed)
    rest = np.sort(rest)
    # units are contiguous codes in both modes.
    va, te = train_test_split(rest, train_size=nval, stratify=unit_y[rest], random_state=seed)
    parts = {name: np.sort(v) if data.groups is None else np.flatnonzero(np.isin(data.groups, v))
             for name, v in zip(("train", "val", "test"), (tr, va, te))}
    outer = np.full(len(y), -1, dtype=np.int64)
    inner = np.full((s["outer_folds"], len(y)), -1, dtype=np.int64)

    def folds(rows, count, fold_seed):
        if data.groups is None:
            if np.bincount(y[rows], minlength=k).min() < count:
                raise ValueError("Insufficient per-class training support for the requested folds")
            splitter = StratifiedKFold(count, shuffle=True, random_state=fold_seed)
            return splitter.split(rows, y[rows])
        for c in range(k):
            if len(np.unique(data.groups[rows[y[rows] == c]])) < count:
                raise ValueError("Insufficient per-class groups for the requested folds")
        splitter = StratifiedGroupKFold(count, shuffle=True, random_state=fold_seed)
        return splitter.split(rows, y[rows], data.groups[rows])

    tr = parts["train"]
    for f, (fi, ho) in enumerate(folds(tr, s["outer_folds"], seed)):
        fit, hold = tr[fi], tr[ho]; outer[hold] = f
        for i, (_, ih) in enumerate(folds(fit, s["inner_folds"], seed + 1000 + f)):
            inner[f, fit[ih]] = i
    for rows in parts.values():
        if np.bincount(y[rows], minlength=k).min() < 1:
            raise ValueError("A split lacks a class; change the split protocol explicitly")
    for f in range(s["outer_folds"]):
        fit = tr[outer[tr] != f]; hold = tr[outer[tr] == f]
        scopes = [fit, hold]
        for i in range(s["inner_folds"]):
            scopes.extend([fit[inner[f, fit] != i], fit[inner[f, fit] == i]])
        if any(np.bincount(y[r], minlength=k).min() == 0 for r in scopes):
            raise ValueError("A nested scope lacks a class; no automatic fold reduction is performed")
        if data.groups is not None:
            if np.intersect1d(data.groups[fit], data.groups[hold]).size:
                raise ValueError("Group overlap across outer folds")
            for i in range(s["inner_folds"]):
                if np.intersect1d(data.groups[fit[inner[f, fit] != i]], data.groups[fit[inner[f, fit] == i]]).size:
                    raise ValueError("Group overlap across inner folds")
    return dict(**parts, outer_fold=outer, inner_fold=inner)


def describe(data, split, cfg):
    from .search import search_budget
    fusion = cfg["dual_view_evidence_fusion"]
    if fusion["enabled"] and (any(c not in data.names for c in fusion["specialized_columns"])
                              or len(fusion["specialized_columns"]) >= len(data.names)):
        raise ValueError("dual_view_evidence_fusion.specialized_columns must be a nonempty proper subset of the selected features")
    classes = len(data.classes); outer = cfg["split"]["outer_folds"]; inner = cfg["split"]["inner_folds"]
    nscopes = 1 + outer + outer * inner
    count = nscopes * (3 + 3 * classes) + ((1 + outer) * 2 * classes if fusion["enabled"] else 0)
    return dict(rows=len(data.y), features=len(data.names), model_columns=2 * len(data.names),
                label=data.label, class_values=data.classes, class_count=classes,
                class_support=np.bincount(data.y, minlength=classes).tolist(), feature_columns=data.names,
                excluded_columns=data.ignored, verified_missing_columns=len(data.checked_flags),
                split_unit="group" if data.groups is not None else "row",
                splits={name: dict(rows=len(split[name]), class_support=np.bincount(data.y[split[name]], minlength=classes).tolist())
                        for name in ("train", "val", "test")},
                native_fit_count=count, dual_view_enabled=fusion["enabled"],
                specialized_columns=fusion["specialized_columns"] if fusion["enabled"] else [],
                search=search_budget(cfg, len(split["train"]), len(split["val"]), classes))
