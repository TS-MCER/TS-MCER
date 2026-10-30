"""CSV + label -> one local all-class OvR experiment; --check-only never fits."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Local CSV with headers")
    parser.add_argument("--label", required=True, help="Target column; every distinct value becomes an OvR class")
    parser.add_argument("--config", type=Path, help="Optional JSON overrides of config.json")
    parser.add_argument("--search", action="store_true", help="Run validation parameter search (enabled by default)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "result",
                        help="New local output directory (default: result beside the source directory)")
    parser.add_argument("--check-only", action="store_true", help="Read/validate inputs and splits; no model import, fit, prediction, or output files")
    args = parser.parse_args(argv)
    try:
        from ovr.config import load_config
        cfg = load_config(args.config)
        if args.search:
            cfg["search"]["enabled"] = True
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(cfg["model"]["cpu_threads"])
        from ovr.data import read_csv, make_split, describe
        data = read_csv(args.csv, args.label, cfg)
        split = make_split(data, cfg)
        info = describe(data, split, cfg)
        if args.check_only:
            print(json.dumps(dict(status="input_check_passed", trained=False, predicted=False, **info), ensure_ascii=False, indent=2))
            return 0
        from ovr.artifacts import digest
        from ovr.pipeline import run_pipeline
        print(f"Starting all-class OvR: {info['class_count']} classes, {info['native_fit_count']} local fits, dual_view_evidence_fusion={info['dual_view_enabled']}", flush=True)
        result = run_pipeline(data, split, cfg, args.output, input_sha256=digest(args.csv),
                              progress=lambda line: print(line, flush=True))
        print(json.dumps(dict(status=result["status"], result="result.json", test=result["metrics"]["test"]), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        if isinstance(error, OSError):
            message = "Input/weight is missing or output exists; check the local paths and use a new output directory"
        else:
            message = str(error)
        print("ERROR: " + message, file=sys.stderr)
        return 2
    except (RuntimeError, TimeoutError) as error:
        print(f"ERROR: {type(error).__name__}; run stopped. Completed artifacts remain in the requested output directory.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
