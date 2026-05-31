"""all_test.py — run every main.py menu choice on a good + bad device.

Enumerates the Cartesian product of the selectable axes (model × dataset ×
augmentation × device) and trains each combination, writing every result
(accuracy, energy, fitter parameters) to a CSV (and JSON) table. Optionally
also sweeps the V3.0 mode toggles (pair / online / sigma_d2d) with
--sweep-modes.

The good/bad device LTP+LTD characteristic xlsx paths are supplied by you:

    python all_test.py --good path/to/good.xlsx --bad path/to/bad.xlsx

Results are written incrementally, so the run is safe to interrupt (Ctrl-C)
and the partial CSV is preserved.

This reuses webapp/gradio_app/runners.run_quick — the same compute path the
web app and main.py use, verified bit-for-bit by tests/parity_check.py — so
numbers match what you'd get by running main.py manually.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
import sys
import time
from datetime import datetime

# Make simulator/ (config, models, ...) and webapp/ (gradio_app) importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "simulator"), os.path.join(_HERE, "webapp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

import config
from gradio_app import schema
from gradio_app import runners


# ---------------------------------------------------------------------------
# Default axes — the choices main.py exposes
# ---------------------------------------------------------------------------
# main.py's 7 models (SimpleNet here follows config.USE_BATCHNORM, like main.py).
DEFAULT_MODELS = [
    "SimpleNet", "Simple_CNN", "Standard_CNN", "LeNet5",
    "VGG", "AlexNet", "ResNet18",
]
DEFAULT_DATASETS = list(schema.DATASET_CHOICES)        # 6 datasets
HEAVY_MODELS = schema.HEAVY_MODELS                     # VGG / AlexNet / ResNet18

# Mode-toggle axes, only enumerated with --sweep-modes
PAIR_AXIS = [(False, "mixed"), (True, "mixed"), (True, "ltp_only")]
ONLINE_AXIS = [False, True]
D2D_AXIS = [0.0, 0.1]

CSV_COLUMNS = [
    "device_label", "device_path", "model", "dataset", "augment",
    "pair_mode", "pair_strategy", "online", "sigma_d2d",
    "epochs", "batch_size", "lr", "psf",
    "final_train_acc", "final_test_acc", "best_test_acc",
    "A_LTP", "A_LTD", "B_LTP", "B_LTD", "sigma_c2c",
    "total_pulses", "total_energy_J", "write_energy_J", "read_energy_J",
    "array_area_m2", "num_weights", "duration_s", "error",
]


def build_combinations(args):
    """Yield one knobs-overlay dict per combination, in run order."""
    devices = [("good", args.good), ("bad", args.bad)]

    if args.augment == "on":
        augs = [True]
    elif args.augment == "off":
        augs = [False]
    else:
        augs = [False, True]

    if args.sweep_modes:
        pair_axis, online_axis, d2d_axis = PAIR_AXIS, ONLINE_AXIS, D2D_AXIS
    else:
        pair_axis, online_axis, d2d_axis = [(False, "mixed")], [False], [0.0]

    for dev_label, dev_path in devices:
        for model in args.models:
            for dataset in args.datasets:
                for aug in augs:
                    for (pair_mode, pair_strategy) in pair_axis:
                        for online in online_axis:
                            for d2d in d2d_axis:
                                yield {
                                    "device_label": dev_label,
                                    "device_path": dev_path,
                                    "model_name": model,
                                    "dataset_name": dataset,
                                    "asl_augment": aug,
                                    "pair_mode": pair_mode,
                                    "pair_strategy": pair_strategy,
                                    "online": online,
                                    "sigma_d2d": d2d,
                                }


def make_knobs(combo, args):
    """Build a full run_quick knobs dict from defaults + this combo."""
    k = schema.defaults()
    k["model_name"] = combo["model_name"]
    k["dataset_name"] = combo["dataset_name"]
    k["device_xlsx"] = combo["device_path"]
    k["asl_augment"] = combo["asl_augment"]
    k["pair_mode"] = combo["pair_mode"]
    k["pair_strategy"] = combo["pair_strategy"]
    k["online"] = combo["online"]
    k["sigma_d2d"] = combo["sigma_d2d"]
    k["epochs"] = args.epochs
    k["max_train_batches"] = args.max_train_batches
    k["seed"] = args.seed
    # SimpleNet follows config.USE_BATCHNORM, matching main.py
    k["use_batchnorm"] = bool(config.USE_BATCHNORM)
    return k


def row_from_result(combo, args, out):
    fit = out["fitter"]
    e = out["energy_report"]
    best = max((h["test_acc"] for h in out["history"]), default=0.0)
    dur = sum(h["secs"] for h in out["history"])
    return {
        "device_label": combo["device_label"],
        "device_path": combo["device_path"],
        "model": combo["model_name"],
        "dataset": combo["dataset_name"],
        "augment": combo["asl_augment"],
        "pair_mode": combo["pair_mode"],
        "pair_strategy": combo["pair_strategy"],
        "online": combo["online"],
        "sigma_d2d": combo["sigma_d2d"],
        "epochs": args.epochs,
        "batch_size": int(schema.defaults()["batch_size"]),
        "lr": float(schema.defaults()["learning_rate"]),
        "psf": float(schema.defaults()["pulse_scaling_factor"]),
        "final_train_acc": round(out["final_train_acc"], 4),
        "final_test_acc": round(out["final_test_acc"], 4),
        "best_test_acc": round(best, 4),
        "A_LTP": fit["A_LTP"], "A_LTD": fit["A_LTD"],
        "B_LTP": fit["B_LTP"], "B_LTD": fit["B_LTD"],
        "sigma_c2c": fit["sigma_c2c"],
        "total_pulses": e["total_pulses"],
        "total_energy_J": e["total_energy_J"],
        "write_energy_J": e["write_energy_J"],
        "read_energy_J": e["read_energy_J"],
        "array_area_m2": e["array_area_m2"],
        "num_weights": e["num_weights"],
        "duration_s": round(dur, 2),
        "error": "",
    }


def error_row(combo, args, msg):
    return {
        **{c: "" for c in CSV_COLUMNS},
        "device_label": combo["device_label"],
        "device_path": combo["device_path"],
        "model": combo["model_name"],
        "dataset": combo["dataset_name"],
        "augment": combo["asl_augment"],
        "pair_mode": combo["pair_mode"],
        "pair_strategy": combo["pair_strategy"],
        "online": combo["online"],
        "sigma_d2d": combo["sigma_d2d"],
        "epochs": args.epochs,
        "error": msg,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run every main.py menu choice on a good + bad device.")
    ap.add_argument("--good", required=True, help="good device LTP/LTD xlsx path")
    ap.add_argument("--bad", required=True, help="bad device LTP/LTD xlsx path")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated subset (default: all 7 main.py models)")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="comma-separated subset (default: all 6 datasets)")
    ap.add_argument("--augment", choices=["both", "on", "off"], default="both",
                    help="enumerate augmentation on+off (both), or fix it")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-modes", action="store_true",
                    help="also enumerate pair/online/sigma_d2d (×12)")
    ap.add_argument("--max-train-batches", type=int, default=0,
                    help="cap batches per epoch for quick smoke runs (0 = no cap)")
    ap.add_argument("--out", default=None, help="results CSV path")
    ap.add_argument("--dry-run", action="store_true",
                    help="list combinations + count, then exit")
    args = ap.parse_args()

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # Validate model names against what the runner understands.
    unknown = [m for m in args.models if m not in schema.MODEL_CHOICES]
    if unknown:
        ap.error(f"unknown model(s): {unknown}. valid: {schema.MODEL_CHOICES}")
    unknown_d = [d for d in args.datasets if d not in schema.DATASET_CHOICES]
    if unknown_d:
        ap.error(f"unknown dataset(s): {unknown_d}. valid: {schema.DATASET_CHOICES}")

    combos = list(build_combinations(args))
    total = len(combos)

    # --- pre-run report ---
    print("=" * 64)
    print("all_test — combination plan")
    print("=" * 64)
    print(f"  devices    : good={args.good}")
    print(f"               bad ={args.bad}")
    print(f"  models     : {args.models}")
    print(f"  datasets   : {args.datasets}")
    print(f"  augment    : {args.augment}")
    print(f"  sweep-modes: {args.sweep_modes}")
    print(f"  epochs     : {args.epochs}"
          + (f"   (max_train_batches={args.max_train_batches})" if args.max_train_batches else ""))
    print(f"  TOTAL RUNS : {total}")
    heavy = sorted(set(args.models) & HEAVY_MODELS)
    if heavy:
        print(f"  ⚠ heavy models on CPU (slow, esp. on CIFAR-100): {heavy}")
    if any(d.upper() == "ASL" for d in args.datasets):
        print("  ⚠ ASL requires datasets/sign_mnist_*.csv; missing → logged as error")
    print("=" * 64)

    if args.dry_run:
        for i, c in enumerate(combos, 1):
            print(f"  [{i:>4}/{total}] {c['device_label']:>4}  {c['model_name']:<13} "
                  f"{c['dataset_name']:<14} aug={c['asl_augment']!s:<5} "
                  f"pair={c['pair_mode']!s:<5}/{c['pair_strategy']:<8} "
                  f"online={c['online']!s:<5} d2d={c['sigma_d2d']}")
        print(f"\n(dry-run) {total} runs would execute. Nothing was trained.")
        return 0

    out_path = args.out or os.path.join(
        _HERE, "results", f"all_test_{datetime.now():%Y%m%d_%H%M%S}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json_path = os.path.splitext(out_path)[0] + ".json"

    rows = []
    t_start = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for i, combo in enumerate(combos, 1):
            tag = (f"[{i}/{total}] {combo['device_label']:>4} "
                   f"{combo['model_name']:<13} {combo['dataset_name']:<14} "
                   f"aug={combo['asl_augment']!s:<5}")
            t0 = time.time()
            try:
                knobs = make_knobs(combo, args)
                out = runners.run_quick(knobs, make_figures=False)
                row = row_from_result(combo, args, out)
                status = (f"train {row['final_train_acc']:.2f}%  "
                          f"test {row['final_test_acc']:.2f}%")
            except Exception as e:  # keep going; record the failure
                row = error_row(combo, args, str(e))
                status = f"ERROR: {str(e)[:60]}"

            writer.writerow(row)
            f.flush()
            rows.append(row)
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(rows, jf, indent=2, default=float)

            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = elapsed / i * (total - i)
            print(f"{tag} → {status}   ({dt:.0f}s, ETA {eta/60:.1f}m)")

    print("\n" + "=" * 64)
    print(f"DONE: {total} runs in {(time.time()-t_start)/60:.1f} min")
    print(f"  CSV : {out_path}")
    print(f"  JSON: {json_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
