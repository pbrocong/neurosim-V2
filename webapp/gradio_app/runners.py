"""Programmatic wrappers around train / test / fitter / optimizer.

These functions accept a flat dict of knob values (keyed by schema.Knob.key)
and return result dicts + matplotlib Figures. No `input()`, no `plt.show()`,
no `subprocess` — everything that breaks under Gradio is gone.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as _project_config
from data_loader import (
    read_split_by_write_vh, get_loaders, get_asl_loaders, get_class_names,
)
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer
from models import (
    SimpleNet, Simple_CNN, Standard_CNN, ResNet18_MNIST,
    VGG_MNIST, AlexNet_MNIST, LeNet5_MNIST,
)
from train_eval import train, train_online, test


# ---------------------------------------------------------------------------
# Model registry: name -> (factory, loss_mode_hint)
# ---------------------------------------------------------------------------
def _model_factory(model_name: str, use_bn: bool):
    """Return (factory(wmin,wmax,nc) -> nn.Module, loss_mode_hint)."""
    if model_name == "SimpleNet":
        return (lambda wmin, wmax, nc: SimpleNet(wmin, wmax, use_bn=bool(use_bn), num_classes=nc),
                "nll")
    if model_name == "SimpleNetBN":
        return (lambda wmin, wmax, nc: SimpleNet(wmin, wmax, use_bn=True, num_classes=nc),
                "nll")
    if model_name == "Simple_CNN":
        return (lambda wmin, wmax, nc: Simple_CNN(wmin, wmax, num_classes=nc), "ce")
    if model_name == "Standard_CNN":
        return (lambda wmin, wmax, nc: Standard_CNN(wmin, wmax, num_classes=nc), "ce")
    if model_name == "LeNet5":
        return (lambda wmin, wmax, nc: LeNet5_MNIST(wmin, wmax, num_classes=nc), "ce")
    if model_name == "VGG":
        return (lambda wmin, wmax, nc: VGG_MNIST(wmin, wmax, num_classes=nc), "ce")
    if model_name == "AlexNet":
        return (lambda wmin, wmax, nc: AlexNet_MNIST(wmin, wmax, num_classes=nc), "ce")
    if model_name == "ResNet18":
        return (lambda wmin, wmax, nc: ResNet18_MNIST(wmin, wmax, num_classes=nc), "ce")
    raise ValueError(f"Unknown model: {model_name!r}")


def _load_dataset(name: str, batch_size: int, balanced_test: bool, asl_augment: bool):
    if name.upper() == "ASL":
        train_loader, test_loader, _ = get_asl_loaders(batch_size, augment=bool(asl_augment))
        if train_loader is None:
            raise FileNotFoundError(
                "ASL CSVs not found. Place sign_mnist_train.csv and "
                "sign_mnist_test.csv under datasets/."
            )
        return train_loader, test_loader, 24
    train_loader, test_loader, nc = get_loaders(
        name, batch_size, balanced_test=bool(balanced_test),
        augment=bool(asl_augment),
    )
    return train_loader, test_loader, nc


def _maybe_subsample_loader(loader, max_batches: int):
    if max_batches is None or max_batches <= 0:
        return loader

    class _Limited:
        def __init__(self, base, k):
            self._base = base
            self._k = int(k)
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


# ---------------------------------------------------------------------------
# Build fitter / optimizer from knobs (shared by run_quick + run_sweep)
# ---------------------------------------------------------------------------
def _build_fitter(knobs: dict, xlsx_path: str) -> NeuroSimFitter:
    ltp, ltd = read_split_by_write_vh(xlsx_path)
    if ltp is None or ltd is None:
        raise RuntimeError(f"Failed to read device data from {xlsx_path}")
    n_states = int(knobs["num_conductance_states"]) if knobs["use_discretisation"] else None
    return NeuroSimFitter(
        ltp, ltd,
        target_range=(float(knobs["target_min"]), float(knobs["target_max"])),
        ltp_fit_ratio=float(knobs["ltp_fit_ratio"]),
        ltd_fit_ratio=float(knobs["ltd_fit_ratio"]),
        sigma_c2c=None,
        sigma_d2d=float(knobs["sigma_d2d"]),
        num_conductance_states=n_states,
        verbose=False,
    )


def _resolve_loss_mode(requested, default_loss: str) -> str:
    """'auto' / None / '' → the model's natural loss; else honor the request."""
    if requested in (None, "", "auto"):
        return default_loss
    return str(requested)


def _energy_params_from_knobs(knobs: dict) -> dict:
    return {
        "E_pulse_LTP_J": float(knobs["E_pulse_LTP_J"]),
        "E_pulse_LTD_J": float(knobs["E_pulse_LTD_J"]),
        "E_read_J":      float(knobs["E_read_J"]),
        "cell_area_m2":  float(knobs["cell_area_m2"]),
    }


# ---------------------------------------------------------------------------
# Quick Run
# ---------------------------------------------------------------------------
def run_quick(knobs: dict, progress_cb=None, make_figures: bool = True) -> dict:
    """One full training run with the given knobs.

    Returns a dict containing history, energy report, and (optionally)
    matplotlib figures. `progress_cb(epoch, total_epochs, status_str)` is
    called once per epoch. `make_figures=False` skips figure generation —
    useful for large batch sweeps (e.g. all_test.py) where only the numbers
    are kept; the "figures" key is then an empty dict.
    """
    torch.manual_seed(int(knobs["seed"]))
    np.random.seed(int(knobs["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xlsx_path = knobs["device_xlsx"] or _project_config.DEFAULT_DEVICE_XLSX
    fitter = _build_fitter(knobs, xlsx_path)

    train_loader, test_loader, num_classes = _load_dataset(
        knobs["dataset_name"],
        int(knobs["batch_size"]),
        bool(knobs["balanced_test"]),
        bool(knobs["asl_augment"]),
    )
    train_loader_eff = _maybe_subsample_loader(train_loader, int(knobs.get("max_train_batches", 0)))

    factory, default_loss = _model_factory(knobs["model_name"], bool(knobs["use_batchnorm"]))
    w_min, w_max = float(knobs["target_min"]), float(knobs["target_max"])
    model = factory(w_min, w_max, num_classes).to(device)

    loss_mode = _resolve_loss_mode(knobs.get("loss_mode"), default_loss)

    optimizer = NeuroSimOptimizer(
        model.parameters(),
        lr=float(knobs["learning_rate"]),
        fitter=fitter,
        pulse_scaling_factor=float(knobs["pulse_scaling_factor"]),
        use_c2c_noise=bool(knobs["use_c2c_noise"]),
        use_discretisation=bool(knobs["use_discretisation"]),
        pair_mode=bool(knobs["pair_mode"]),
        pair_strategy=str(knobs["pair_strategy"]),
        energy_params=_energy_params_from_knobs(knobs),
        d2d_seed=int(knobs["d2d_seed"]),
    )

    history = []
    epochs = int(knobs["epochs"])
    online = bool(knobs["online"])
    online_cap = int(knobs["online_max_samples"]) or None

    preds_last, targets_last = None, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if online:
            tr_acc = train_online(
                model, device, train_loader_eff, optimizer, ep,
                loss_mode=loss_mode,
                micro_batch=int(knobs["online_micro_batch"]),
                max_samples_per_epoch=online_cap,
                progress_every=0,
            )
        else:
            tr_acc = train(model, device, train_loader_eff, optimizer, ep, loss_mode=loss_mode)
        if ep == epochs:
            te_acc, preds_last, targets_last = test(
                model, device, test_loader, loss_mode=loss_mode, return_preds=True,
            )
        else:
            te_acc = test(model, device, test_loader, loss_mode=loss_mode)
        dt = time.time() - t0
        history.append({"epoch": ep, "train_acc": float(tr_acc),
                        "test_acc": float(te_acc), "secs": float(dt)})
        if progress_cb is not None:
            progress_cb(ep, epochs,
                        f"epoch {ep}/{epochs}  train {tr_acc:.2f}%  test {te_acc:.2f}%")

    energy = optimizer.energy_report()

    if make_figures:
        figs = {
            "accuracy": _fig_accuracy(history),
            "confusion": _fig_confusion(preds_last, targets_last,
                                        get_class_names(knobs["dataset_name"], num_classes)),
        }
    else:
        figs = {}
    return {
        "history": history,
        "final_train_acc": history[-1]["train_acc"] if history else 0.0,
        "final_test_acc":  history[-1]["test_acc"] if history else 0.0,
        "fitter": {
            "A_LTP": fitter.A_LTP, "A_LTD": fitter.A_LTD,
            "B_LTP": fitter.B_LTP, "B_LTD": fitter.B_LTD,
            "sigma_c2c": fitter.sigma_c2c,
            "Pmax_LTP": fitter.Pmax_LTP, "Pmax_LTD": fitter.Pmax_LTD,
        },
        "energy_report": energy,
        "figures": figs,
        "preds": preds_last,
        "targets": targets_last,
        "num_classes": int(num_classes),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Sweep (cross-product like verification_suite)
# ---------------------------------------------------------------------------
def run_sweep(knobs: dict, progress_cb=None) -> dict:
    """Run cross-product (model × dataset) of small training cells.

    Reuses the common hyperparameters from `knobs`; the lists of models /
    datasets / epoch-per-cell come from sweep_*-keyed knobs.
    """
    models_csv = str(knobs["sweep_models"])
    datasets_csv = str(knobs["sweep_datasets"])
    epochs_cell = int(knobs["sweep_epochs"])
    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    datasets = [d.strip() for d in datasets_csv.split(",") if d.strip()]

    cells = [(m, d) for m in models for d in datasets]
    rows: list[dict] = []
    total = len(cells)
    for i, (m, d) in enumerate(cells, start=1):
        cell_knobs = dict(knobs)
        cell_knobs["model_name"] = m
        cell_knobs["dataset_name"] = d
        cell_knobs["epochs"] = epochs_cell
        try:
            out = run_quick(cell_knobs)
            rows.append({
                "model": m, "dataset": d, "epochs": epochs_cell,
                "final_train_acc": out["final_train_acc"],
                "final_test_acc":  out["final_test_acc"],
                "best_test_acc": max(h["test_acc"] for h in out["history"]),
                "A_LTP": out["fitter"]["A_LTP"],
                "A_LTD": out["fitter"]["A_LTD"],
                "sigma_c2c": out["fitter"]["sigma_c2c"],
                "duration_s": sum(h["secs"] for h in out["history"]),
                "error": "",
            })
        except Exception as e:
            rows.append({
                "model": m, "dataset": d, "epochs": epochs_cell,
                "final_train_acc": 0.0, "final_test_acc": 0.0,
                "best_test_acc": 0.0,
                "A_LTP": 0.0, "A_LTD": 0.0, "sigma_c2c": 0.0,
                "duration_s": 0.0,
                "error": str(e),
            })
        if progress_cb is not None:
            progress_cb(i, total, f"cell {i}/{total}: {m} × {d}")

    return {"rows": rows, "figure": _fig_sweep(rows)}


# ---------------------------------------------------------------------------
# Energy (no training — pure arithmetic from ENERGY_PARAMS)
# ---------------------------------------------------------------------------
def compute_energy(knobs: dict) -> dict:
    """Reproduce neurosim_utils.energy_report() arithmetic without training.

    Uses the user's assumed n_samples and per-mode pulse counts, plus the
    number of weights derived from instantiating the chosen model.
    """
    factory, _ = _model_factory(knobs["model_name"], bool(knobs["use_batchnorm"]))
    # Build a placeholder model just to count weights; instantiate on CPU.
    nc = 24 if knobs["dataset_name"] == "ASL" else (
        100 if knobs["dataset_name"] == "CIFAR-100" else 10
    )
    w_min, w_max = float(knobs["target_min"]), float(knobs["target_max"])
    model = factory(w_min, w_max, nc)
    total_weights = int(sum(p.numel() for p in model.parameters()))

    ep = _energy_params_from_knobs(knobs)
    n_samples = int(knobs["energy_n_samples"])
    p_ltp = int(knobs["energy_pulses_ltp"])
    p_ltd = int(knobs["energy_pulses_ltd"])
    pair = bool(knobs["pair_mode"])

    e_write = p_ltp * ep["E_pulse_LTP_J"] + p_ltd * ep["E_pulse_LTD_J"]
    n_reads = n_samples * total_weights
    e_read = n_reads * ep["E_read_J"]
    area = total_weights * ep["cell_area_m2"] * (2 if pair else 1)

    report = {
        "model": knobs["model_name"],
        "dataset": knobs["dataset_name"],
        "num_weights": total_weights,
        "pair_mode": pair,
        "total_pulses_ltp": float(p_ltp),
        "total_pulses_ltd": float(p_ltd),
        "total_pulses":     float(p_ltp + p_ltd),
        "total_reads":      int(n_reads),
        "write_energy_J":   float(e_write),
        "read_energy_J":    float(e_read),
        "total_energy_J":   float(e_write + e_read),
        "array_area_m2":    float(area),
    }
    report["figure"] = _fig_energy_bars(report)
    return report


# ---------------------------------------------------------------------------
# Degradation (port of degradation_ui.run_one minus Streamlit)
# ---------------------------------------------------------------------------
def run_degradation(knobs: dict, progress_cb=None) -> dict:
    """Train once on normal, then apply degraded mask, train again, compare.

    This is intentionally a minimal port: ratio_single only on the first pass
    so we land a working tab; ratio_sweep is exposed in the schema for the
    user to drive multiple invocations.
    """
    rng = np.random.default_rng(int(knobs["deg_seed"]))
    torch.manual_seed(int(knobs["seed"]))
    np.random.seed(int(knobs["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal_fitter = _build_fitter(knobs, knobs["deg_normal_xlsx"])
    degraded_fitter = _build_fitter(knobs, knobs["deg_degraded_xlsx"])

    train_loader, test_loader, num_classes = _load_dataset(
        knobs["dataset_name"],
        int(knobs["batch_size"]),
        bool(knobs["balanced_test"]),
        bool(knobs["asl_augment"]),
    )

    factory, default_loss = _model_factory(knobs["model_name"], bool(knobs["use_batchnorm"]))
    w_min, w_max = float(knobs["target_min"]), float(knobs["target_max"])
    model = factory(w_min, w_max, num_classes).to(device)
    loss_mode = _resolve_loss_mode(knobs.get("loss_mode"), default_loss)

    target_layers = [s.strip() for s in str(knobs["deg_target_layers"]).split(",") if s.strip()]

    # Build masks (cells marked True will use the degraded fitter)
    masks = {}
    target_param_ids = []
    for name, p in model.named_parameters():
        # match by top-level layer name (e.g. "fc1", "fc2", "conv1")
        head = name.split(".")[0]
        if head in target_layers and p.requires_grad and p.dim() >= 2:
            target_param_ids.append(id(p))
            masks[id(p)] = _build_mask(p.shape, knobs, rng)

    optimizer = NeuroSimOptimizer(
        model.parameters(),
        lr=float(knobs["learning_rate"]),
        fitter=normal_fitter,
        pulse_scaling_factor=float(knobs["pulse_scaling_factor"]),
        degraded_fitter=degraded_fitter,
        masks=masks,
        use_c2c_noise=bool(knobs["use_c2c_noise"]),
        use_discretisation=bool(knobs["use_discretisation"]),
        pair_mode=bool(knobs["pair_mode"]),
        pair_strategy=str(knobs["pair_strategy"]),
        energy_params=_energy_params_from_knobs(knobs),
        d2d_seed=int(knobs["d2d_seed"]),
    )

    history = []
    epochs = int(knobs["epochs"])
    for ep_i in range(1, epochs + 1):
        if bool(knobs["online"]):
            tr_acc = train_online(
                model, device, train_loader, optimizer, ep_i,
                loss_mode=loss_mode,
                micro_batch=int(knobs["online_micro_batch"]),
                max_samples_per_epoch=int(knobs["online_max_samples"]) or None,
                progress_every=0,
            )
        else:
            tr_acc = train(model, device, train_loader, optimizer, ep_i, loss_mode=loss_mode)
        te_acc = test(model, device, test_loader, loss_mode=loss_mode)
        history.append({"epoch": ep_i, "train_acc": float(tr_acc), "test_acc": float(te_acc)})
        if progress_cb is not None:
            progress_cb(ep_i, epochs,
                        f"epoch {ep_i}/{epochs}  train {tr_acc:.2f}%  test {te_acc:.2f}%")

    # Per-layer degraded counts
    mask_stats = {
        name: {"num_degraded": int(masks[id(p)].sum().item()) if id(p) in masks else 0,
               "total": int(p.numel())}
        for name, p in model.named_parameters() if p.dim() >= 2
    }
    return {
        "history": history,
        "final_train_acc": history[-1]["train_acc"] if history else 0.0,
        "final_test_acc":  history[-1]["test_acc"] if history else 0.0,
        "mask_stats": mask_stats,
        "target_layers": target_layers,
        "mode": knobs["deg_mode"],
        "figures": {"accuracy": _fig_accuracy(history)},
    }


def _build_mask(shape, knobs: dict, rng: np.random.Generator) -> torch.Tensor:
    """Return a bool tensor of the same shape marking degraded cells."""
    flat = torch.zeros(int(np.prod(shape)), dtype=torch.bool)
    mode = knobs["deg_mode"]
    n = flat.numel()
    if mode in ("ratio_single", "ratio_sweep"):
        ratio = float(knobs["deg_ratio_single"]) / 100.0
        k = int(round(ratio * n))
        if k > 0:
            picks = rng.choice(n, size=k, replace=False)
            flat[picks] = True
    elif mode == "range":
        a, b = int(knobs["deg_range_start"]), int(knobs["deg_range_end"])
        a = max(0, min(a, n)); b = max(0, min(b, n))
        flat[a:b] = True
    elif mode == "indices":
        idx_csv = str(knobs["deg_indices"]).strip()
        if idx_csv:
            for tok in idx_csv.split(","):
                tok = tok.strip()
                if tok.isdigit():
                    i = int(tok)
                    if 0 <= i < n:
                        flat[i] = True
    return flat.view(*shape)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _fig_accuracy(history) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if history:
        xs = [h["epoch"] for h in history]
        ax.plot(xs, [h["train_acc"] for h in history], marker="o", label="train")
        ax.plot(xs, [h["test_acc"]  for h in history], marker="s", label="test")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no history", ha="center", va="center")
    fig.tight_layout()
    return fig


def _fig_confusion(preds, targets, class_names) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if preds is None or targets is None or len(preds) == 0:
        ax.text(0.5, 0.5, "no predictions", ha="center", va="center")
        return fig
    n_cls = max(max(targets) + 1, max(preds) + 1, len(class_names))
    m = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(targets, preds):
        m[int(t), int(p)] += 1
    ax.imshow(m, cmap="Blues", aspect="auto")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    if n_cls <= 24:
        ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
        ax.set_xticklabels(class_names[:n_cls], rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names[:n_cls], fontsize=7)
    fig.tight_layout()
    return fig


def _fig_sweep(rows) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r['model']}\n{r['dataset']}" for r in rows]
    ys = [r["best_test_acc"] for r in rows]
    ax.bar(range(len(rows)), ys, color="#3a78c2")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("best test acc (%)")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_energy_bars(rep) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cats = ["Write", "Read"]
    vals = [rep["write_energy_J"] * 1e6, rep["read_energy_J"] * 1e6]
    bars = ax.bar(cats, vals, color=["#d05050", "#3a78c2"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f} µJ",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Energy (µJ)")
    ax.set_title(f"{rep['model']}  ({rep['num_weights']:,} weights"
                 f"{', pair' if rep['pair_mode'] else ''})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
