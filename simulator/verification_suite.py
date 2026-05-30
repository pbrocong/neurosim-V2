# verification_suite.py
#
# Cross-product sweep of (model, dataset) to verify that after the V3.0 refactor:
#   1. MLP on MNIST hits ≥90 % on the ZnO_Encap_Oven_48h_2 (linear) device.
#   2. CNNs reach literature-level accuracy on MNIST (95+ %).
#   3. CNN accuracy on ASL is dataset-limited (overfit gap), not optimizer-limited.
#
# Saves a JSON report next to the script and a CSV summary.

from __future__ import annotations
import argparse, json, os, sys, time, warnings, csv
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

import config
from data_loader import read_split_by_write_vh, get_mnist_loaders, get_asl_loaders
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer
from models import SimpleNet, Simple_CNN, Standard_CNN, LeNet5_MNIST
from train_eval import train, test


# (model_name, model_factory, loss_mode, num_classes_overrideable)
MODELS = {
    "SimpleNet":    lambda nc: SimpleNet(-1, 1, use_bn=True, num_classes=nc),
    "Simple_CNN":   lambda nc: Simple_CNN(-1, 1, num_classes=nc),
    "Standard_CNN": lambda nc: Standard_CNN(-1, 1, num_classes=nc),
    "LeNet5":       lambda nc: LeNet5_MNIST(-1, 1, num_classes=nc),
}
LOSS_BY_MODEL = {"SimpleNet": "nll", "Simple_CNN": "ce",
                 "Standard_CNN": "ce", "LeNet5": "ce"}


def run_one(model_name: str, dataset: str, epochs: int, lr: float, psf: float,
            asl_augment: bool = False, seed: int = 0) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ltp, ltd = read_split_by_write_vh(config.DEFAULT_DEVICE_XLSX)
    fitter = NeuroSimFitter(ltp, ltd, target_range=(-1, 1), verbose=False)

    if dataset == "MNIST":
        train_loader, test_loader = get_mnist_loaders(128)
        nc = 10
    elif dataset == "ASL":
        train_loader, test_loader, _ = get_asl_loaders(128, augment=asl_augment)
        nc = 24
    else:
        raise ValueError(dataset)

    model = MODELS[model_name](nc)
    opt = NeuroSimOptimizer(model.parameters(), lr=lr, fitter=fitter,
                            pulse_scaling_factor=psf, use_c2c_noise=True)
    loss_mode = LOSS_BY_MODEL[model_name]

    t0 = time.time()
    accs = []
    train_accs = []
    for ep in range(1, epochs + 1):
        ta = train(model, "cpu", train_loader, opt, ep, loss_mode=loss_mode)
        te = test(model, "cpu", test_loader, loss_mode=loss_mode)
        train_accs.append(round(float(ta), 2))
        accs.append(round(float(te), 2))
    dt = time.time() - t0

    return {
        "model": model_name, "dataset": dataset,
        "epochs": epochs, "lr": lr, "psf": psf,
        "asl_augment": asl_augment,
        "A_LTP": fitter.A_LTP, "A_LTD": fitter.A_LTD,
        "B_LTP": fitter.B_LTP, "B_LTD": fitter.B_LTD,
        "sigma_c2c": fitter.sigma_c2c,
        "Pmax_LTP": fitter.Pmax_LTP, "Pmax_LTD": fitter.Pmax_LTD,
        "train_acc_per_epoch": train_accs,
        "test_acc_per_epoch": accs,
        "best_test_acc": max(accs) if accs else 0.0,
        "final_test_acc": accs[-1] if accs else 0.0,
        "duration_s": round(dt, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3,
                    help="epochs per cell (kept small for sweep speed)")
    ap.add_argument("--out", default="verification_results.json")
    args = ap.parse_args()

    pairs = [
        # (model, dataset, lr, psf, augment)
        ("SimpleNet",    "MNIST", 1e-2, 5, False),
        ("SimpleNet",    "ASL",   1e-2, 5, False),
        ("Simple_CNN",   "MNIST", 1e-2, 5, False),
        ("Simple_CNN",   "ASL",   1e-2, 5, False),
        ("Simple_CNN",   "ASL",   5e-3, 10, True),
        ("Standard_CNN", "MNIST", 1e-2, 5, False),
        ("Standard_CNN", "ASL",   1e-2, 5, False),
        ("LeNet5",       "MNIST", 1e-2, 5, False),
        ("LeNet5",       "ASL",   1e-2, 5, False),
    ]

    results = []
    for (model, dataset, lr, psf, aug) in pairs:
        print(f"\n=== {model}  {dataset}  lr={lr}  psf={psf}  aug={aug} ===")
        try:
            r = run_one(model, dataset, args.epochs, lr, psf, asl_augment=aug)
            print(f"  -> best={r['best_test_acc']:.2f}% final={r['final_test_acc']:.2f}%  "
                  f"({r['duration_s']:.0f}s)")
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"model": model, "dataset": dataset,
                            "lr": lr, "psf": psf, "error": str(e)})
        # checkpoint after each
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)
    # CSV summary too
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "lr", "psf", "aug", "epochs",
                    "best_test_acc", "final_test_acc",
                    "A_LTP", "A_LTD", "sigma_c2c"])
        for r in results:
            if "error" in r: continue
            w.writerow([r["model"], r["dataset"], r["lr"], r["psf"],
                        r["asl_augment"], r["epochs"],
                        r["best_test_acc"], r["final_test_acc"],
                        round(r["A_LTP"], 3), round(r["A_LTD"], 3),
                        round(r["sigma_c2c"], 4)])
    print(f"\nSaved: {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
