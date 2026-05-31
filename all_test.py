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
import matplotlib.pyplot as plt
import numpy as np

import config
from data_loader import get_class_names
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


# ---------------------------------------------------------------------------
# Interactive device-path input
# ---------------------------------------------------------------------------
def _clean_path(s: str) -> str:
    """Tidy a pasted/dragged path: strip quotes, whitespace, escaped spaces."""
    s = (s or "").strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = s.replace("\\ ", " ").strip()
    return os.path.expanduser(s)


def resolve_device_path(val, label):
    """Return a valid path: use `val` if given+exists, else prompt until valid."""
    val = _clean_path(val) if val else ""
    while not (val and os.path.exists(val)):
        if val:
            print(f"  ⚠ 파일을 찾을 수 없습니다: {val}")
        try:
            val = _clean_path(input(f"{label} 경로를 입력하세요: "))
        except (EOFError, KeyboardInterrupt):
            print("\n취소되었습니다.")
            sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# Per-run figures (saved into the organized output folder)
# ---------------------------------------------------------------------------
def save_accuracy_fig(history, path, title):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if history:
        xs = [h["epoch"] for h in history]
        ax.plot(xs, [h["train_acc"] for h in history], marker="o", label="train")
        ax.plot(xs, [h["test_acc"] for h in history], marker="s", label="test")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_confusion_fig(preds, targets, class_names, path, title):
    """Row-normalized confusion matrix labelled with class names."""
    n = len(class_names)
    if not preds or not targets or n == 0:
        return
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)

    size = max(5.0, min(2.0 + n * 0.42, 16.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    rot = 0 if all(len(str(c)) <= 2 for c in class_names) else 45
    ax.set_xticklabels(class_names, rotation=rot, ha="right" if rot else "center", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=9)
    if n <= 20:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_subdir(out_dir, combo):
    """results/<ts>/<device>/<model>/<dataset>_aug-<on|off>[_<modes>]/"""
    name = f"{combo['dataset_name']}_aug-{'on' if combo['asl_augment'] else 'off'}"
    if combo["pair_mode"]:
        name += f"_pair-{combo['pair_strategy']}"
    if combo["online"]:
        name += "_online"
    if combo["sigma_d2d"]:
        name += f"_d2d{combo['sigma_d2d']}"
    d = os.path.join(out_dir, combo["device_label"], combo["model_name"], name)
    os.makedirs(d, exist_ok=True)
    return d


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
    ap.add_argument("--good", default=None, help="good device LTP/LTD xlsx path "
                    "(omit to be prompted)")
    ap.add_argument("--bad", default=None, help="bad device LTP/LTD xlsx path "
                    "(omit to be prompted)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated subset (default: all 7 main.py models)")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help="comma-separated subset (default: all 6 datasets)")
    ap.add_argument("--augment", choices=["both", "on", "off"], default="both",
                    help="enumerate augmentation on+off (both), or fix it")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-modes", action="store_true",
                    help="also enumerate pair/online/sigma_d2d (×12)")
    ap.add_argument("--max-train-batches", type=int, default=0,
                    help="cap batches per epoch for quick smoke runs (0 = no cap)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: results/all_test_<timestamp>/)")
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

    # Ask for the good / bad device paths interactively if not supplied
    # (skipped for --dry-run, which only counts combinations).
    if args.dry_run:
        args.good = args.good or "<good.xlsx>"
        args.bad = args.bad or "<bad.xlsx>"
    else:
        print("\n--- 소자 특성 파일 경로 입력 ---")
        args.good = resolve_device_path(args.good, "좋은 소자 (good) LTP/LTD xlsx")
        args.bad = resolve_device_path(args.bad, "나쁜 소자 (bad)  LTP/LTD xlsx")

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

    out_dir = args.out or os.path.join(
        _HERE, "results", f"all_test_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    json_path = os.path.join(out_dir, "summary.json")
    print(f"\n결과 폴더: {out_dir}\n")

    rows = []
    t_start = time.time()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
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
                # per-run artifacts, organized by category
                sub = run_subdir(out_dir, combo)
                ttl = (f"{combo['device_label']} · {combo['model_name']} · "
                       f"{combo['dataset_name']} · aug={combo['asl_augment']}")
                save_accuracy_fig(out["history"], os.path.join(sub, "accuracy.png"), ttl)
                save_confusion_fig(out.get("preds"), out.get("targets"),
                                   get_class_names(combo["dataset_name"], out["num_classes"]),
                                   os.path.join(sub, "confusion.png"), ttl)
                with open(os.path.join(sub, "metrics.json"), "w", encoding="utf-8") as mf:
                    json.dump(row, mf, indent=2, default=float)
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

    summary_txt = write_grouped_summary(rows, out_dir)

    print("\n" + "=" * 64)
    print(f"DONE: {total} runs in {(time.time()-t_start)/60:.1f} min")
    print(f"  폴더        : {out_dir}")
    print(f"  요약 표     : summary.csv / summary.json")
    print(f"  분류별 요약 : summary.txt")
    print(f"  각 실행별   : <device>/<model>/<dataset>_aug-*/{{accuracy,confusion}}.png + metrics.json")
    print("=" * 64)
    print(summary_txt)
    return 0


def write_grouped_summary(rows, out_dir):
    """Aggregate final_test_acc by device / model / dataset and write summary.txt."""
    ok = [r for r in rows if not r.get("error")]

    def agg(key):
        groups = {}
        for r in ok:
            groups.setdefault(r[key], []).append(float(r["final_test_acc"]))
        lines = []
        for k in sorted(groups):
            vs = groups[k]
            lines.append(f"    {str(k):<16} mean={sum(vs)/len(vs):6.2f}%  "
                         f"max={max(vs):6.2f}%  min={min(vs):6.2f}%  (n={len(vs)})")
        return "\n".join(lines)

    parts = ["=" * 64, "분류별 정확도 요약 (final test accuracy)", "=" * 64,
             f"  성공 {len(ok)} / 전체 {len(rows)} runs", ""]
    for key, title in [("device_label", "소자별 (good vs bad)"),
                       ("model", "모델별"),
                       ("dataset", "데이터셋별"),
                       ("augment", "augmentation별")]:
        parts.append(f"  [{title}]")
        parts.append(agg(key) or "    (없음)")
        parts.append("")
    errs = [r for r in rows if r.get("error")]
    if errs:
        parts.append(f"  [에러 {len(errs)}건]")
        for r in errs[:20]:
            parts.append(f"    {r['device_label']}/{r['model']}/{r['dataset']} "
                         f"aug={r['augment']}: {str(r['error'])[:60]}")
        parts.append("")
    text = "\n".join(parts)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
