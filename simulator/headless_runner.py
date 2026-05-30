# headless_runner.py
#
# Non-interactive driver for verification sweeps. Trains a chosen model on a
# chosen dataset with the NeuroSim device characteristic from a given xlsx,
# prints per-epoch train/test accuracy and returns the final test accuracy.
#
# Intended for autonomous "sweep + table" workflows (the GUI / input() in
# main.py blocks that automation).

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import io

# Suppress matplotlib in headless runs.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

import config
from data_loader import (
    read_split_by_write_vh, get_loaders, get_asl_loaders, DATASET_REGISTRY,
)
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer
from models import (
    SimpleNet, Simple_CNN, Standard_CNN, ResNet18_MNIST,
    VGG_MNIST, AlexNet_MNIST, LeNet5_MNIST,
)
from train_eval import train, train_online, test


MODEL_REGISTRY = {
    "SimpleNet":   (lambda wmin, wmax, nc: SimpleNet(wmin, wmax, use_bn=False, num_classes=nc), "nll"),
    "SimpleNetBN": (lambda wmin, wmax, nc: SimpleNet(wmin, wmax, use_bn=True, num_classes=nc), "nll"),
    "Simple_CNN":  (lambda wmin, wmax, nc: Simple_CNN(wmin, wmax, num_classes=nc), "ce"),
    "Standard_CNN":(lambda wmin, wmax, nc: Standard_CNN(wmin, wmax, num_classes=nc), "ce"),
    "LeNet5":      (lambda wmin, wmax, nc: LeNet5_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "VGG":         (lambda wmin, wmax, nc: VGG_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "AlexNet":     (lambda wmin, wmax, nc: AlexNet_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "ResNet18":    (lambda wmin, wmax, nc: ResNet18_MNIST(wmin, wmax, num_classes=nc), "ce"),
}


def load_dataset(dataset_name: str, batch_size: int):
    if dataset_name.upper() == "ASL":
        train_loader, test_loader, _td = get_asl_loaders(batch_size)
        if train_loader is None:
            raise FileNotFoundError("ASL CSVs not found")
        return train_loader, test_loader, 24
    train_loader, test_loader, num_classes = get_loaders(
        dataset_name, batch_size, balanced_test=False
    )
    return train_loader, test_loader, num_classes


def _maybe_subsample_loader(loader, max_batches):
    """Wrap a DataLoader so it only yields up to `max_batches` batches.

    Used for quick smoke runs; preserves shuffling because we iterate from the
    underlying loader each call.
    """
    if max_batches is None:
        return loader

    class _Limited:
        def __init__(self, base, k):
            self._base = base
            self._k = k
            try:
                self.dataset = base.dataset
            except AttributeError:
                self.dataset = None

        def __iter__(self):
            for i, batch in enumerate(self._base):
                if i >= self._k:
                    return
                yield batch

        def __len__(self):
            return min(self._k, len(self._base))

    return _Limited(loader, max_batches)


def run(model_name: str,
        dataset_name: str,
        device_xlsx: str | None,
        epochs: int,
        lr: float,
        psf: float,
        batch_size: int,
        seed: int = 0,
        use_c2c: bool = True,
        use_disc: bool = False,
        num_states: int | None = None,
        max_train_batches: int | None = None,
        # V3.0 advanced modes
        pair_mode: bool = False,
        pair_strategy: str = "mixed",
        sigma_d2d: float = 0.0,
        online: bool = False,
        online_micro_batch: int = 1,
        online_max_samples: int | None = None,
        verbose_fit: bool = False) -> dict:
    """Run a single training and return final accuracies + energy report."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_xlsx is None:
        device_xlsx = config.DEFAULT_DEVICE_XLSX
    ltp, ltd = read_split_by_write_vh(device_xlsx)
    if ltp is None or ltd is None:
        raise RuntimeError(f"Failed to read device data from {device_xlsx}")

    fitter = NeuroSimFitter(
        ltp, ltd,
        target_range=config.TARGET_RANGE,
        sigma_c2c=None,                         # auto
        sigma_d2d=float(sigma_d2d),
        num_conductance_states=num_states,
        verbose=verbose_fit,
    )

    train_loader, test_loader, num_classes = load_dataset(dataset_name, batch_size)
    train_loader_eff = _maybe_subsample_loader(train_loader, max_train_batches)

    factory, loss_mode = MODEL_REGISTRY[model_name]
    w_min, w_max = config.TARGET_RANGE
    model = factory(w_min, w_max, num_classes).to(device)

    optimizer = NeuroSimOptimizer(
        model.parameters(), lr=lr, fitter=fitter, pulse_scaling_factor=psf,
        use_c2c_noise=use_c2c, use_discretisation=use_disc,
        pair_mode=bool(pair_mode),
        pair_strategy=str(pair_strategy),
        energy_params=getattr(config, "ENERGY_PARAMS", None),
        d2d_seed=int(seed),
    )

    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if online:
            tr_acc = train_online(
                model, device, train_loader_eff, optimizer, ep,
                loss_mode=loss_mode,
                micro_batch=int(online_micro_batch),
                max_samples_per_epoch=online_max_samples,
                progress_every=0,
            )
        else:
            tr_acc = train(model, device, train_loader_eff, optimizer, ep, loss_mode=loss_mode)
        te_acc = test(model, device, test_loader, loss_mode=loss_mode)
        dt = time.time() - t0
        history.append({"epoch": ep, "train_acc": tr_acc, "test_acc": te_acc, "secs": dt})

    final = history[-1] if history else {"train_acc": 0.0, "test_acc": 0.0}
    energy = optimizer.energy_report()
    return {
        "model": model_name,
        "dataset": dataset_name,
        "device_xlsx": device_xlsx,
        "epochs": epochs,
        "lr": lr, "psf": psf, "batch_size": batch_size, "seed": seed,
        "pair_mode": bool(pair_mode),
        "pair_strategy": str(pair_strategy),
        "sigma_d2d": float(sigma_d2d),
        "online": bool(online),
        "online_micro_batch": int(online_micro_batch),
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "history": history,
        "A_LTP": fitter.A_LTP, "A_LTD": fitter.A_LTD,
        "B_LTP": fitter.B_LTP, "B_LTD": fitter.B_LTD,
        "sigma_c2c": fitter.sigma_c2c,
        "Pmax_LTP": fitter.Pmax_LTP, "Pmax_LTD": fitter.Pmax_LTD,
        "energy_report": energy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device-xlsx", default=None)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--psf", type=float, default=config.PULSE_SCALING_FACTOR)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-c2c", action="store_true")
    ap.add_argument("--discretise", action="store_true")
    ap.add_argument("--num-states", type=int, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None,
                    help="cap batches per epoch (smoke runs)")
    # V3.0 advanced modes
    ap.add_argument("--pair-mode", action="store_true",
                    help="differential pair (G⁺ − G⁻) mode")
    ap.add_argument("--pair-strategy", choices=["mixed", "ltp_only"], default="mixed",
                    help="pair update rule: mixed (V3.0 canonical) or ltp_only")
    ap.add_argument("--sigma-d2d", type=float, default=0.0,
                    help="device-to-device σ on A (log-normal); 0 = off")
    ap.add_argument("--online", action="store_true",
                    help="online (batch=1) training mode; BN → eval")
    ap.add_argument("--online-micro-batch", type=int, default=1,
                    help="samples per update in online mode (1=pure)")
    ap.add_argument("--online-max-samples", type=int, default=None,
                    help="cap samples per epoch in online mode")
    ap.add_argument("--energy-report", action="store_true",
                    help="print energy/area report at the end")

    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = run(args.model, args.dataset, args.device_xlsx,
              args.epochs, args.lr, args.psf, args.batch_size, args.seed,
              use_c2c=not args.no_c2c,
              use_disc=args.discretise,
              num_states=args.num_states,
              max_train_batches=args.max_train_batches,
              pair_mode=args.pair_mode,
              pair_strategy=args.pair_strategy,
              sigma_d2d=args.sigma_d2d,
              online=args.online,
              online_micro_batch=args.online_micro_batch,
              online_max_samples=args.online_max_samples,
              verbose_fit=True)
    if args.json:
        print(json.dumps(out, indent=2, default=float))
    else:
        print(f"\n=== FINAL  model={args.model}  dataset={args.dataset}  "
              f"pair={args.pair_mode}  d2d={args.sigma_d2d}  online={args.online}")
        print(f"     test={out['final_test_acc']:.2f}%  train={out['final_train_acc']:.2f}% ===")
        if args.energy_report:
            r = out["energy_report"]
            print("\n--- Energy Report ---")
            print(f"  LTP pulses  : {r['total_pulses_ltp']:.0f}")
            print(f"  LTD pulses  : {r['total_pulses_ltd']:.0f}")
            print(f"  Total pulses: {r['total_pulses']:.0f}")
            print(f"  Reads       : {r['total_reads']:,}")
            print(f"  Write energy: {r['write_energy_J']*1e6:.4f} µJ")
            print(f"  Read energy : {r['read_energy_J']*1e6:.4f} µJ")
            print(f"  Total energy: {r['total_energy_J']*1e6:.4f} µJ")
            print(f"  Area        : {r['array_area_m2']*1e12:.2f} µm² (pair={r['pair_mode']})")


if __name__ == "__main__":
    main()
