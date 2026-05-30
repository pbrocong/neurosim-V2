"""Parity harness: Gradio path (gradio_app.runners) vs original VSCode path
(headless_runner.run).

Both call the same compute primitives (train/test/NeuroSimFitter/
NeuroSimOptimizer). With identical seed + hyperparameters the final
accuracies and the energy report must match bit-for-bit. Any divergence
means the deployed HF app would behave differently from the local code.

Run:  python tests/parity_check.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import sys

# simulator/ holds the core modules (config, headless_runner, ...);
# webapp/ holds the gradio_app package. Add both to the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "simulator"), os.path.join(_ROOT, "webapp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

import config
import headless_runner as HR
from gradio_app import schema
from gradio_app import runners


# Tiny, fast, fully deterministic budget.
COMMON = dict(epochs=1, batch_size=64, seed=0, max_train_batches=5)

# Each case: (name, hl_model, gr_model, use_bn, loss_mode, dataset, adv)
# `adv` overrides advanced modes on BOTH paths identically.
# loss_mode="auto" proves the UI default resolves to each model's natural loss
# and therefore reproduces the VSCode entry points out of the box.
def C(name, hl, gr, bn, loss, ds, **adv):
    return (name, hl, gr, bn, loss, ds, adv)


CASES = [
    # --- basic (default modes) ---
    C("SimpleNet  (auto→nll, no BN)", "SimpleNet",   "SimpleNet",   False, "auto", "MNIST"),
    C("SimpleNetBN (auto→nll, BN)",   "SimpleNetBN", "SimpleNet",   True,  "auto", "MNIST"),
    C("Simple_CNN (auto→ce)",         "Simple_CNN",  "Simple_CNN",  False, "auto", "MNIST"),
    C("LeNet5     (auto→ce)",         "LeNet5",      "LeNet5",      False, "auto", "MNIST"),
    C("Standard_CNN (auto→ce)",       "Standard_CNN","Standard_CNN",False, "auto", "MNIST"),
    # --- advanced modes ---
    C("pair_mode mixed",     "Simple_CNN", "Simple_CNN", False, "auto", "MNIST",
      pair_mode=True, pair_strategy="mixed"),
    C("pair_mode ltp_only",  "Simple_CNN", "Simple_CNN", False, "auto", "MNIST",
      pair_mode=True, pair_strategy="ltp_only"),
    C("sigma_d2d=0.10",      "SimpleNet",  "SimpleNet",  False, "auto", "MNIST",
      sigma_d2d=0.10),
    C("discretise 64 states","Simple_CNN", "Simple_CNN", False, "auto", "MNIST",
      use_disc=True, num_states=64),
    C("online micro_batch=4","SimpleNet",  "SimpleNet",  False, "auto", "MNIST",
      online=True, online_micro_batch=4),
    C("no C2C noise",        "Simple_CNN", "Simple_CNN", False, "auto", "MNIST",
      use_c2c=False),
]


def run_headless(model, dataset, adv):
    return HR.run(
        model_name=model,
        dataset_name=dataset,
        device_xlsx=None,
        epochs=COMMON["epochs"],
        lr=config.LEARNING_RATE,
        psf=config.PULSE_SCALING_FACTOR,
        batch_size=COMMON["batch_size"],
        seed=COMMON["seed"],
        use_c2c=adv.get("use_c2c", True),
        use_disc=adv.get("use_disc", False),
        num_states=adv.get("num_states", None),
        max_train_batches=COMMON["max_train_batches"],
        pair_mode=adv.get("pair_mode", False),
        pair_strategy=adv.get("pair_strategy", "mixed"),
        sigma_d2d=adv.get("sigma_d2d", 0.0),
        online=adv.get("online", False),
        online_micro_batch=adv.get("online_micro_batch", 1),
        online_max_samples=adv.get("online_max_samples", None),
        verbose_fit=False,
    )


def run_gradio(model, dataset, use_bn, loss_mode, adv):
    k = schema.defaults()
    k.update(COMMON)
    k["model_name"] = model
    k["dataset_name"] = dataset
    k["use_batchnorm"] = use_bn
    k["loss_mode"] = loss_mode
    # match headless defaults explicitly
    k["learning_rate"] = config.LEARNING_RATE
    k["pulse_scaling_factor"] = config.PULSE_SCALING_FACTOR
    k["target_min"], k["target_max"] = config.TARGET_RANGE
    # advanced overrides (note: headless seeds d2d with `seed`, so mirror that)
    k["use_c2c_noise"] = adv.get("use_c2c", True)
    k["use_discretisation"] = adv.get("use_disc", False)
    if adv.get("num_states") is not None:
        k["num_conductance_states"] = adv["num_states"]
    k["pair_mode"] = adv.get("pair_mode", False)
    k["pair_strategy"] = adv.get("pair_strategy", "mixed")
    k["sigma_d2d"] = adv.get("sigma_d2d", 0.0)
    k["d2d_seed"] = COMMON["seed"]          # headless uses seed as d2d_seed
    k["online"] = adv.get("online", False)
    k["online_micro_batch"] = adv.get("online_micro_batch", 1)
    k["online_max_samples"] = adv.get("online_max_samples", 0) or 0
    return runners.run_quick(k)


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main():
    print("=" * 70)
    print("PARITY CHECK  —  Gradio runners  vs  headless_runner")
    print("=" * 70)
    all_ok = True
    for name, hl_model, gr_model, use_bn, loss_mode, dataset, adv in CASES:
        hl = run_headless(hl_model, dataset, adv)
        gd = run_gradio(gr_model, dataset, use_bn, loss_mode, adv)

        checks = {
            "final_train_acc": (hl["final_train_acc"], gd["final_train_acc"]),
            "final_test_acc":  (hl["final_test_acc"],  gd["final_test_acc"]),
            "write_energy_J":  (hl["energy_report"]["write_energy_J"],
                                gd["energy_report"]["write_energy_J"]),
            "read_energy_J":   (hl["energy_report"]["read_energy_J"],
                                gd["energy_report"]["read_energy_J"]),
            "total_pulses":    (hl["energy_report"]["total_pulses"],
                                gd["energy_report"]["total_pulses"]),
            "array_area_m2":   (hl["energy_report"]["array_area_m2"],
                                gd["energy_report"]["array_area_m2"]),
            "A_LTP":           (hl["A_LTP"], gd["fitter"]["A_LTP"]),
            "sigma_c2c":       (hl["sigma_c2c"], gd["fitter"]["sigma_c2c"]),
        }
        case_ok = all(approx(a, b) for a, b in checks.values())
        all_ok = all_ok and case_ok
        status = "✅ MATCH" if case_ok else "❌ MISMATCH"
        print(f"\n[{status}]  {name}")
        for metric, (a, b) in checks.items():
            mark = "  " if approx(a, b) else " ⚠"
            print(f"   {mark} {metric:16s} headless={a:<22} gradio={b}")

    print("\n" + "=" * 70)
    print("RESULT:", "ALL CASES MATCH ✅" if all_ok else "DISCREPANCIES FOUND ❌")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
