"""Synthetic integration checks for development-only operator search."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_pipeline import synthetic_csv, TestBackend, fixed_refinement_config
import numpy as np

from ovr import operators as op
from ovr.artifacts import arrays
from ovr.config import validate
from ovr.data import read_csv, make_split, describe
from ovr.evaluation import score
from ovr.pipeline import run_pipeline
from ovr.search import rank_key


class SearchTests(unittest.TestCase):
    def config(self):
        cfg = fixed_refinement_config()
        cfg["split"].update(outer_folds=2, inner_folds=2)
        cfg["dual_view_evidence_fusion"].update(enabled=True, specialized_columns=["x3", "x4", "x5"])
        cfg["search"].update(enabled=True, consensus_refinement_grid=dict(huber_clip=[.5, 2.], k=[8, 70],
            alpha=[0., .875], class_power=[0., .25], pseudo=[4., 8.]),
            dual_view_evidence_fusion_grid=dict(beta_plus=[0., .5], evidence_clip=[1., 3.], increment_clip=[.5, 2.], tau=[0., .125]))
        return cfg

    def test_search_outputs_selection_and_no_test_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); csv = root / "synthetic.csv"; synthetic_csv(csv, nclass=3)
            cfg = self.config(); d = read_csv(csv, "target", cfg); s = make_split(d, cfg)
            runs = []
            def guard(path):
                self.assertNotIn("test", Path(path).name.lower())
                self.assertFalse((current / "native" / "full_train" / "raw_test.npz").exists())
                return arrays(path)
            for repeat in range(2):
                current = root / f"result_{repeat}"
                altered = copy.deepcopy(d)
                if repeat:
                    altered.y[s["test"]] = (altered.y[s["test"]] + 1) % 3
                backend = TestBackend()
                with patch("ovr.search.arrays", side_effect=guard):
                    result = run_pipeline(altered, s, cfg, current, backend=backend, progress=None)
                self.assertEqual(backend.fit_calls, describe(d, s, cfg)["native_fit_count"])
                selection = json.loads((current / "search" / "selection.json").read_text())
                self.assertFalse(selection["test_used_for_selection"])
                self.assertEqual(len(selection["consensus_candidates"]), 32)
                self.assertEqual(len(selection["fusion_candidates"]), 16)
                for stage, prefix, probability_key in (("consensus_refinement", "consensus", "consensus_refined"),
                                                       ("dual_view_evidence_fusion", "fusion", "p")):
                    archive = arrays(current / "search" / f"{stage}_development.npz")
                    records = selection[f"{prefix}_candidates"]
                    selected = selection[f"selected_{prefix}_index"]
                    independent = min(range(len(records)), key=lambda i: rank_key(
                        score(archive["val_y"], archive["val_p"][i]), records[i]["uid"], cfg["search"]["metric"]))
                    self.assertEqual(selected, independent)
                    for part, key in (("train_oof", "train_oof_p"), ("val", "val_p")):
                        final = arrays(current / "predictions" / f"{part}.npz")
                        np.testing.assert_allclose(final[probability_key], archive[key][selected], atol=1e-14, rtol=0)
                state = arrays(current / "selected_memory" / "full_train" / "state.npz")
                training = arrays(current / "memory" / "full_train" / "training.npz")
                expected = op.build_residual_memory(training, result["effective_config"]["consensus_refinement"]["huber_clip"])
                for key in expected:
                    np.testing.assert_array_equal(state[key], expected[key])
                runs.append((selection, arrays(current / "predictions" / "test.npz")))
            self.assertEqual(runs[0][0], runs[1][0])
            np.testing.assert_array_equal(runs[0][1]["p"], runs[1][1]["p"])

    def test_dual_view_disabled_and_zero_dual_view_evidence_fusion_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); csv = root / "synthetic.csv"; synthetic_csv(csv, nclass=3)
            for enabled in (False, True):
                cfg = self.config(); cfg["dual_view_evidence_fusion"]["enabled"] = enabled
                cfg["search"]["consensus_refinement_grid"] = {k: [v[0]] for k, v in cfg["search"]["consensus_refinement_grid"].items()}
                cfg["search"]["dual_view_evidence_fusion_grid"] = {k: [v[0]] for k, v in cfg["search"]["dual_view_evidence_fusion_grid"].items()}
                d = read_csv(csv, "target", cfg); s = make_split(d, cfg)
                output = root / str(enabled)
                run_pipeline(d, s, cfg, output, backend=TestBackend(), progress=None)
                self.assertEqual((output / "search" / "dual_view_evidence_fusion_development.npz").exists(), enabled)
                for part in ("train_oof", "val", "test"):
                    z = arrays(output / "predictions" / f"{part}.npz")
                    np.testing.assert_array_equal(z["p"], z["consensus_refined"])

    def test_budget_and_bad_grid_fail_before_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); csv = root / "synthetic.csv"; synthetic_csv(csv, nclass=3)
            cfg = self.config(); d = read_csv(csv, "target", cfg); s = make_split(d, cfg)
            cfg["search"]["max_candidates"] = 1
            backend = TestBackend()
            with self.assertRaisesRegex(ValueError, "max_candidates"):
                run_pipeline(d, s, cfg, root / "result", backend=backend, progress=None)
            self.assertEqual(backend.fit_calls, 0)
            self.assertFalse((root / "result").exists())
            for value in ([], [0], [True], [.5, .5]):
                cfg = self.config(); cfg["search"]["consensus_refinement_grid"]["huber_clip"] = value
                with self.assertRaises(ValueError):
                    validate(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
