"""Synthetic-only checks. No repository dataset or model checkpoint is read."""

import copy
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from ovr.artifacts import arrays
from ovr.config import load_config, validate
from ovr.data import read_csv, make_split, describe, FoldPreprocessor
from ovr.experts import contexts, coverage, quota
from ovr import operators as op
from ovr.pipeline import run_pipeline


def fixed_refinement_config():
    """Explicit control configuration for checks without search or fusion."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.json"
        path.write_text(json.dumps({"dual_view_evidence_fusion": {"enabled": False},
                                    "search": {"enabled": False}}), encoding="utf-8")
        return load_config(path)


def synthetic_csv(path, nclass=6, per_class=30):
    """Entirely generated values; all identifiers are fabricated."""
    rng = np.random.default_rng(731)
    y = np.repeat(np.arange(nclass), per_class)
    rng.shuffle(y)
    values = rng.normal(4 + y[:, None] * .3, 1, size=(len(y), 6))
    values[:, 0] = rng.integers(0, 2, size=len(y))
    values[::13, 3] = np.nan
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", *[f"x{i}" for i in range(6)], *[f"x{i}__missing" for i in range(6)], "target"])
        for i, (x, label) in enumerate(zip(values, y)):
            writer.writerow([f"synthetic_{i}", *["?" if np.isnan(v) else v for v in x],
                             *np.isnan(x).astype(int).tolist(), f"class_{label}"])


class TestBackend:
    """Fast test double only: the production CLI cannot select this backend."""
    info = {"backend": "synthetic_test_double", "native_TabPFN": False}

    def __init__(self):
        self.fit_calls = 0

    def fit(self, x, y, seed):
        self.fit_calls += 1
        return LogisticRegression(random_state=seed, max_iter=200).fit(x, y)

    def save(self, model, path):
        joblib.dump(model, path)

    def load(self, path):
        return joblib.load(path)

    def release(self):
        pass


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.csv = self.folder / "synthetic.csv"
        synthetic_csv(self.csv)
        self.cfg = fixed_refinement_config()

    def write(self, text):
        self.csv.write_text(text, encoding="utf-8")

    def test_all_classes_and_single_missing_indicators(self):
        d = read_csv(self.csv, "target", self.cfg)
        self.assertEqual(d.classes, [f"class_{c}" for c in range(6)])
        self.assertEqual(len(d.names), 6)
        self.assertIn("record_id", d.ignored)
        self.assertEqual(len(d.checked_flags), 6)
        self.assertEqual(FoldPreprocessor(d.names).fit(d.x[:90]).transform(d.x).shape, (180, 12))

    def test_class_order_numeric_and_string(self):
        self.write("x,target\n1,10\n2,2\n3,1\n")
        self.assertEqual(read_csv(self.csv, "target", self.cfg).classes, ["1", "2", "10"])
        self.write("x,target\n1,NA_class\n2,A\n")
        self.assertEqual(read_csv(self.csv, "target", self.cfg).classes, ["A", "NA_class"])

    def test_reject_duplicate_headers_unknown_values_and_missing_labels(self):
        for text in ("x,x,target\n1,2,a\n2,3,b\n", "x,target\ntext,a\n1,b\n",
                     "x,target\n1,\n2,a\n", "x,target\ninf,a\n1,b\n"):
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(ValueError):
                    read_csv(self.csv, "target", self.cfg)

    def test_existing_flags_are_checked(self):
        self.write("x,x__missing,target\n?,0,a\n1,0,b\n")
        with self.assertRaisesRegex(ValueError, "disagree"):
            read_csv(self.csv, "target", self.cfg)

    def test_literal_negative_values_need_explicit_missing_contract(self):
        self.write("x,target\n-1,a\n-2,b\n")
        self.cfg["data"]["missing_values"] = [""]
        np.testing.assert_array_equal(read_csv(self.csv, "target", self.cfg).x[:, 0], [-1, -2])

    def test_fold_statistics_do_not_use_heldout_values(self):
        pp = FoldPreprocessor(["a", "b"], ["b"]).fit([[1, 0], [3, 1], [np.nan, 0]])
        np.testing.assert_array_equal(pp.fill, [2, 0])
        np.testing.assert_array_equal(pp.mean, [2, 0])
        np.testing.assert_array_equal(pp.std, [1, 1])
        np.testing.assert_array_equal(pp.transform([[np.nan, np.nan]]), [[0, 0, 1, 1]])
        np.testing.assert_array_equal(pp.transform([[100, 1]]), [[98, 1, 0, 0]])
        all_missing = FoldPreprocessor(["a"], []).fit([[np.nan], [np.nan]])
        np.testing.assert_array_equal(all_missing.transform([[np.nan], [3]]), [[0, 1], [3, 0]])

    def test_nested_split_disjointness_and_repeatability(self):
        d = read_csv(self.csv, "target", self.cfg); s = make_split(d, self.cfg)
        t = make_split(d, self.cfg)
        for key in s:
            np.testing.assert_array_equal(s[key], t[key])
        np.testing.assert_array_equal(np.sort(np.r_[s["train"], s["val"], s["test"]]), np.arange(len(d.y)))
        for f in range(5):
            fit = s["train"][s["outer_fold"][s["train"]] != f]
            hold = s["train"][s["outer_fold"][s["train"]] == f]
            self.assertTrue((s["inner_fold"][f, hold] == -1).all())
            for i in range(4):
                a = fit[s["inner_fold"][f, fit] != i]; b = fit[s["inner_fold"][f, fit] == i]
                self.assertEqual(len(np.intersect1d(a, b)), 0)
                self.assertEqual(len(np.unique(d.y[a])), 6)

    def test_group_isolation(self):
        with self.csv.open("w", newline="", encoding="utf-8") as h:
            w = csv.writer(h); w.writerow(["group", "x", "target"])
            group_labels = np.random.default_rng(731).permutation(np.tile(np.arange(3), 100))
            for g, label in enumerate(group_labels):
                for visit in range(2):
                    w.writerow([f"group_{g}", visit, f"class_{label}"])
        self.cfg["data"]["group_column"] = "group"
        d = read_csv(self.csv, "target", self.cfg); s = make_split(d, self.cfg)
        self.assertNotIn("group", d.names)
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertEqual(len(np.intersect1d(d.groups[s[a]], d.groups[s[b]])), 0)

    def test_bad_group_labels_and_small_classes_fail(self):
        self.write("group,x,target\nsame,1,a\nsame,2,b\n")
        self.cfg["data"]["group_column"] = "group"
        with self.assertRaises(ValueError):
            read_csv(self.csv, "target", self.cfg)
        synthetic_csv(self.csv, per_class=3)
        self.cfg["data"]["group_column"] = None
        with self.assertRaises(ValueError):
            make_split(read_csv(self.csv, "target", self.cfg), self.cfg)

    def test_check_only_no_model_import_and_no_files(self):
        import main
        settings = self.folder / "settings.json"
        settings.write_text(json.dumps({"dual_view_evidence_fusion": {
            "specialized_columns": ["x3", "x4", "x5"]}}), encoding="utf-8")
        with patch("ovr.pipeline.run_pipeline", side_effect=AssertionError("must not train")), patch("builtins.print"):
            self.assertEqual(main.main(["--csv", str(self.csv), "--label", "target",
                                        "--config", str(settings), "--check-only"]), 0)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["settings.json", "synthetic.csv"])

    def test_invalid_configuration_and_specialized_columns(self):
        d = read_csv(self.csv, "target", self.cfg); s = make_split(d, self.cfg)
        self.cfg["dual_view_evidence_fusion"].update(enabled=True, specialized_columns=["absent"])
        with self.assertRaises(ValueError):
            describe(d, s, self.cfg)
        self.cfg["dual_view_evidence_fusion"]["specialized_columns"] = d.names
        with self.assertRaises(ValueError):
            describe(d, s, self.cfg)
        self.cfg["split"]["outer_folds"] = 1
        with self.assertRaises(ValueError):
            validate(self.cfg)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(314)
        self.train = dict(row_index=np.arange(90), y=np.tile(np.arange(6), 15), fold=np.arange(90) % 5,
                          base_probabilities=rng.dirichlet(np.ones(6), 90), context_probabilities=rng.uniform(.02, .98, (90, 6, 3)))
        self.query = dict(row_index=np.arange(90, 110), base_probabilities=rng.dirichlet(np.ones(6), 20),
                          context_probabilities=rng.uniform(.02, .98, (20, 6, 3)))
        self.cfg = fixed_refinement_config(); self.state = op.build_residual_memory(self.train, .5)

    def test_independent_scalar_formula(self):
        p, extra = op.refine_consensus(self.query, self.state, self.cfg["consensus_refinement"])
        # Deliberately scalar: no reuse of risk/geometry/residual implementations.
        reference = np.empty_like(p)
        def logodds(v):
            v = min(max(float(v), 1e-7), 1 - 1e-7)
            return np.log(v) - np.log1p(-v)
        for row in range(len(p)):
            for c in range(6):
                logits = np.asarray([logodds(v) for v in self.query["context_probabilities"][row, c]])
                mu = np.median(logits)
                for _ in range(5):
                    mu += sum(min(max(v - mu, -.5), .5) for v in logits) / 3
                risk = np.asarray([logodds(self.query["base_probabilities"][row, c]), logits.mean()])
                query = np.clip((risk - self.state["mean"][c]) / self.state["std"][c], -5, 5)
                dist = np.asarray([sum((query - v) ** 2) for v in self.state["x"][:, c]])
                nearest = sorted(range(90), key=lambda i: (dist[i], i))[:32]
                truth = self.state["labels"][:, c]
                counts = [sum(truth == t) for t in (0, 1)]
                weight = np.asarray([(90 / (2 * counts[int(t)])) ** .25 for t in truth])
                prior = sum(weight * truth) / sum(weight)
                w = np.asarray([np.exp(-dist[i] / max(dist[nearest[-1]], 1e-8)) * weight[i] for i in nearest])
                w /= sum(w); ess = 1 / sum(w ** 2)
                obs = (ess * sum(w * truth[nearest]) + 8 * prior) / (ess + 8)
                exp = (ess * sum(w * self.state["qsource"][nearest, c]) + 8 * prior) / (ess + 8)
                delta = min(max(logodds(obs) - logodds(exp), -2), 2)
                reference[row, c] = expit(mu + .75 * delta)
        reference /= reference.sum(1, keepdims=True)
        np.testing.assert_allclose(p, reference, atol=2e-14, rtol=0)

    def test_row_order_subbatch_and_no_query_memory_leak(self):
        p, _ = op.refine_consensus(self.query, self.state, self.cfg["consensus_refinement"])
        for take in (np.arange(20)[::-1], np.arange(7)):
            sub, _ = op.refine_consensus({k: v[take] for k, v in self.query.items()}, self.state, self.cfg["consensus_refinement"])
            np.testing.assert_allclose(sub, p[take], atol=2e-14, rtol=0)
        self.query["row_index"][0] = 0
        with self.assertRaises(ValueError):
            op.refine_consensus(self.query, self.state, self.cfg["consensus_refinement"])

    def test_dual_view_formula_disable_and_subbatch(self):
        b, _ = op.refine_consensus(self.query, self.state, self.cfg["consensus_refinement"])
        path = self.query["context_probabilities"][:, :, 0]; full = self.query["context_probabilities"][:, :, 1]
        result, increment = op.fuse_dual_view_evidence(b, path, full, self.cfg["dual_view_evidence_fusion"])
        reference = expit(op.lg(b) + np.clip(.375 * np.maximum(np.clip(op.lg(path) - op.lg(full), -3, 3), 0), -2, 2))
        reference /= reference.sum(1, keepdims=True); reference[np.all(increment == 0, axis=1)] = b[np.all(increment == 0, axis=1)]
        np.testing.assert_allclose(result, reference, atol=2e-14, rtol=0)
        np.testing.assert_array_equal(op.fuse_dual_view_evidence(b, path, full, dict(self.cfg["dual_view_evidence_fusion"], beta_plus=0))[0], b)
        np.testing.assert_array_equal(op.fuse_dual_view_evidence(b, path, path, self.cfg["dual_view_evidence_fusion"])[0], b)
        np.testing.assert_allclose(op.fuse_dual_view_evidence(b[::-1], path[::-1], full[::-1], self.cfg["dual_view_evidence_fusion"])[0], result[::-1], atol=2e-14, rtol=0)

    def test_large_k_and_binary_class_generalization(self):
        cfg = dict(self.cfg["consensus_refinement"], k=10000)
        p, _ = op.refine_consensus(self.query, self.state, cfg)
        np.testing.assert_allclose(p.sum(1), 1)
        t = {k: v for k, v in self.train.items()}
        t["y"] = t["y"] % 2; t["base_probabilities"] = op.normalize(t["base_probabilities"][:, :2]); t["context_probabilities"] = t["context_probabilities"][:, :2]
        q = dict(self.query, base_probabilities=op.normalize(self.query["base_probabilities"][:, :2]), context_probabilities=self.query["context_probabilities"][:, :2])
        p, _ = op.refine_consensus(q, op.build_residual_memory(t, .5), cfg)
        self.assertEqual(p.shape, (20, 2))

    def test_sampling_keeps_all_positives_and_no_duplicates(self):
        y = self.train["y"]; rows = np.arange(90); x = np.random.default_rng(4).normal(size=(90, 12))
        selected = list(contexts(y, rows, x, 6, 6, 42, 0))
        self.assertEqual(len(selected), 18)
        for c, _, context in selected:
            self.assertEqual(len(context), len(np.unique(context)))
            self.assertTrue(np.isin(rows[y == c], context).all())
            self.assertEqual(len(np.unique(y[context])), 6)
        np.testing.assert_array_equal(quota([1, 2, 10], 8), [1, 1, 6])
        np.testing.assert_array_equal(coverage(np.arange(5), np.zeros((5, 2)), 3, 1), [0, 1, 2])


class EndToEndTests(unittest.TestCase):
    def test_consensus_refinement_only_uses_every_class_and_query_labels_cannot_change_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); csv_path = folder / "synthetic.csv"
            synthetic_csv(csv_path, nclass=3)
            cfg = fixed_refinement_config(); cfg["split"].update(outer_folds=2, inner_folds=2)
            d = read_csv(csv_path, "target", cfg); split = make_split(d, cfg)
            first = run_pipeline(d, split, cfg, folder / "before", backend=TestBackend(), progress=None)
            self.assertEqual(first["native_fit_count"], 84)
            changed = copy.deepcopy(d)
            for part in ("val", "test"):
                changed.y[split[part]] = np.roll(changed.y[split[part]], 1)
            second = run_pipeline(changed, split, cfg, folder / "after", backend=TestBackend(), progress=None)
            self.assertEqual(second["native_fit_count"], 84)
            for part in ("train_oof", "val", "test"):
                a = arrays(folder / "before" / "predictions" / f"{part}.npz")
                b = arrays(folder / "after" / "predictions" / f"{part}.npz")
                np.testing.assert_array_equal(a["p"], a["consensus_refined"])
                np.testing.assert_array_equal(a["p"], b["p"])
                self.assertEqual(a["p"].shape[1], 3)
            self.assertFalse(any((folder / "before" / "native").rglob("specialized_view_*")))

    def test_default_nested_protocol_dual_view_and_output_reuse_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); csv_path = folder / "synthetic.csv"; synthetic_csv(csv_path)
            settings = folder / "settings.json"
            settings.write_text(json.dumps({"dual_view_evidence_fusion": {
                "specialized_columns": ["x3", "x4", "x5"]}}), encoding="utf-8")
            cfg = load_config(settings)
            self.assertTrue(cfg["search"]["enabled"])
            self.assertTrue(cfg["dual_view_evidence_fusion"]["enabled"])
            d = read_csv(csv_path, "target", cfg); split = make_split(d, cfg)
            backend = TestBackend()
            with patch.dict(os.environ):
                result = run_pipeline(d, split, cfg, folder / "result", backend=backend, progress=None)
            self.assertEqual(backend.fit_calls, 618)
            self.assertEqual(result["native_fit_count"], 618)
            self.assertTrue(result["verification"]["strict_nested_oof"])
            selection = json.loads((folder / "result" / "search" / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selection["consensus_candidates"]), 384)
            self.assertEqual(len(selection["fusion_candidates"]), 54)
            self.assertFalse(selection["test_used_for_selection"])
            for part in ("train_oof", "val", "test"):
                z = arrays(folder / "result" / "predictions" / f"{part}.npz")
                self.assertEqual(z["p"].shape[1], 6)
                self.assertNotIn("record_id", z)
                np.testing.assert_allclose(z["p"].sum(1), 1)
            for f in range(5):
                t = arrays(folder / "result" / "memory" / f"outer_{f}" / "training.npz")
                held = split["train"][split["outer_fold"][split["train"]] == f]
                self.assertEqual(len(np.intersect1d(held, t["row_index"])), 0)
            with self.assertRaises(FileExistsError):
                run_pipeline(d, split, cfg, folder / "result", backend=backend, progress=None)

    def test_standalone_copy_and_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "standalone"
            shutil.copytree(ROOT, package, ignore=shutil.ignore_patterns("result", "__pycache__", ".git"))
            data = Path(tmp) / "synthetic.csv"; synthetic_csv(data)
            settings = Path(tmp) / "settings.json"
            settings.write_text(json.dumps({"dual_view_evidence_fusion": {
                "specialized_columns": ["x3", "x4", "x5"]}}), encoding="utf-8")
            completed = subprocess.run([sys.executable, "-B", str(package / "main.py"), "--csv", str(data),
                                        "--label", "target", "--config", str(settings), "--check-only"], cwd=tmp,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["class_count"], 6)
            self.assertTrue(json.loads(completed.stdout)["dual_view_enabled"])
            self.assertFalse((package / "result").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
