"""Single-source-of-truth metadata for every NeuroSim hyperparameter.

Each Knob describes one configurable value: where its default lives, what
widget to draw, its valid range, and which parent knob (if any) enables it.

`widgets.py` iterates this list to build the UI. `config_io.py` uses it to
validate JSON. `runners.py` reads from the dict the widgets produce, keyed
by `Knob.key`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import config as _project_config


# ---------------------------------------------------------------------------
# Knob dataclass
# ---------------------------------------------------------------------------
@dataclass
class Knob:
    key: str
    label: str
    category: str
    widget: str                                # checkbox|slider|number|dropdown|text|range
    dtype: type
    default: Any
    bounds: Optional[tuple] = None             # (min, max) or (min, max, step)
    choices: Optional[list] = None
    depends_on: Optional[dict] = None          # {"PARENT_KEY": expected_value}
    source: str = ""                           # "config.py:46" trace
    help: str = ""                             # tooltip


# ---------------------------------------------------------------------------
# Model & dataset registries (mirrors headless_runner.MODEL_REGISTRY +
# data_loader.DATASET_REGISTRY but kept here so widgets can read it without
# importing torch at schema-load time)
# ---------------------------------------------------------------------------
MODEL_CHOICES = [
    "SimpleNet", "SimpleNetBN", "Simple_CNN", "Standard_CNN",
    "LeNet5", "VGG", "AlexNet", "ResNet18",
]

# Models known to be heavy on HF Spaces CPU — surface as a warning in the UI.
HEAVY_MODELS = {"VGG", "AlexNet", "ResNet18"}

DATASET_CHOICES = [
    "MNIST", "Fashion-MNIST", "K-MNIST", "CIFAR-10", "CIFAR-100", "ASL",
]

PAIR_STRATEGY_CHOICES = ["mixed", "ltp_only"]
# "auto" lets each model use its natural loss (SimpleNet→nll, CNNs→ce),
# matching the VSCode entry points. Explicit ce/nll overrides it.
LOSS_MODE_CHOICES = ["auto", "ce", "nll"]
DEGRADATION_MODE_CHOICES = ["ratio_sweep", "ratio_single", "range", "indices"]


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------
_ENERGY = _project_config.ENERGY_PARAMS

KNOBS: list[Knob] = [
    # ----- Model & Dataset selection -----
    Knob("model_name", "Model architecture", "Model",
         widget="dropdown", dtype=str, default="SimpleNet",
         choices=MODEL_CHOICES,
         help="Network to train. VGG/AlexNet/ResNet18 are heavy on CPU."),
    Knob("dataset_name", "Dataset", "Dataset",
         widget="dropdown", dtype=str, default="MNIST",
         choices=DATASET_CHOICES,
         help="ASL requires sign_mnist_*.csv to be present in datasets/."),

    # ----- Training -----
    Knob("epochs", "Epochs", "Training",
         widget="slider", dtype=int, default=int(_project_config.EPOCHS),
         bounds=(1, 50, 1),
         source="config.py:21",
         help="Number of full passes over the training set."),
    Knob("batch_size", "Batch size", "Training",
         widget="number", dtype=int, default=int(_project_config.BATCH_SIZE),
         bounds=(1, 1024, 1),
         source="config.py:20"),
    Knob("learning_rate", "Learning rate", "Training",
         widget="number", dtype=float, default=float(_project_config.LEARNING_RATE),
         bounds=(1e-6, 1.0, None),
         source="config.py:22",
         help="Scales pulse counts; effective step depends also on PSF."),
    Knob("pulse_scaling_factor", "Pulse scaling factor (PSF)", "Training",
         widget="number", dtype=float, default=float(_project_config.PULSE_SCALING_FACTOR),
         bounds=(0.1, 100.0, None),
         source="config.py:23"),
    Knob("seed", "RNG seed", "Training",
         widget="number", dtype=int, default=0, bounds=(0, 2**31 - 1, 1),
         help="Sets torch + numpy + random seeds for reproducibility."),
    Knob("loss_mode", "Loss mode", "Training",
         widget="dropdown", dtype=str, default="auto",
         choices=LOSS_MODE_CHOICES,
         help="'auto' uses each model's natural loss (SimpleNet→nll, CNNs→ce). "
              "Override with 'nll' (log_softmax output) or 'ce' (logits)."),

    # ----- Optimizer / pulse domain -----
    Knob("target_min", "Weight target min", "Optimizer",
         widget="number", dtype=float, default=float(_project_config.TARGET_RANGE[0]),
         bounds=(-10.0, 0.0, None),
         source="config.py:24"),
    Knob("target_max", "Weight target max", "Optimizer",
         widget="number", dtype=float, default=float(_project_config.TARGET_RANGE[1]),
         bounds=(0.0, 10.0, None),
         source="config.py:24"),
    Knob("ltp_fit_ratio", "LTP fit ratio", "Fitter",
         widget="slider", dtype=float, default=float(_project_config.LTP_FIT_RATIO),
         bounds=(0.05, 1.0, 0.05),
         source="config.py:27",
         help="Fraction of LTP data used for curve fitting."),
    Knob("ltd_fit_ratio", "LTD fit ratio", "Fitter",
         widget="slider", dtype=float, default=float(_project_config.LTD_FIT_RATIO),
         bounds=(0.05, 1.0, 0.05),
         source="config.py:28"),

    # ----- Device non-idealities -----
    Knob("use_c2c_noise", "Cycle-to-cycle noise", "Device",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_C2C_NOISE),
         source="config.py:41",
         help="σ_c2c noise added per pulse update."),
    Knob("use_discretisation", "Conductance discretisation", "Device",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_CONDUCTANCE_DISCRETISATION),
         source="config.py:42",
         help="Quantize G to a finite number of states."),
    Knob("num_conductance_states", "Num conductance states", "Device",
         widget="number", dtype=int,
         default=int(_project_config.NUM_CONDUCTANCE_STATES or 64),
         bounds=(2, 4096, 1),
         depends_on={"use_discretisation": True},
         source="config.py:43",
         help="e.g. 64 = 6-bit weight."),
    Knob("sigma_d2d", "σ_d2d (device-to-device)", "Device",
         widget="number", dtype=float,
         default=float(_project_config.SIGMA_D2D),
         bounds=(0.0, 2.0, None),
         source="config.py:49",
         help="Log-normal variation on A. 0 = identical cells."),
    Knob("d2d_seed", "D2D RNG seed", "Device",
         widget="number", dtype=int,
         default=int(_project_config.D2D_SEED),
         bounds=(0, 2**31 - 1, 1),
         source="config.py:50"),

    # ----- Pair mode -----
    Knob("pair_mode", "Differential pair (G⁺ − G⁻)", "Pair",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_PAIR_MODE),
         source="config.py:46"),
    Knob("pair_strategy", "Pair update strategy", "Pair",
         widget="dropdown", dtype=str,
         default=str(_project_config.PAIR_STRATEGY),
         choices=PAIR_STRATEGY_CHOICES,
         depends_on={"pair_mode": True},
         source="config.py:47",
         help="'mixed' = V3.0 canonical; 'ltp_only' = Burr/Boybat variant."),

    # ----- Online training -----
    Knob("online", "Online (batch-1) training", "Online",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_ONLINE_TRAINING),
         source="config.py:53",
         help="Forces BatchNorm to eval mode."),
    Knob("online_micro_batch", "Online micro-batch", "Online",
         widget="number", dtype=int,
         default=int(_project_config.ONLINE_MICRO_BATCH),
         bounds=(1, 64, 1),
         depends_on={"online": True},
         source="config.py:54",
         help="1 = pure online; 4-8 = almost-online for speed."),
    Knob("online_max_samples", "Online max samples/epoch", "Online",
         widget="number", dtype=int,
         default=int(_project_config.ONLINE_MAX_SAMPLES_PER_EPOCH or 0),
         bounds=(0, 1_000_000, 1),
         depends_on={"online": True},
         source="config.py:55",
         help="0 = no cap. Useful for smoke runs."),

    # ----- Model architecture -----
    Knob("use_batchnorm", "Use BatchNorm (SimpleNet only)", "Model",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_BATCHNORM),
         source="config.py:38",
         help="If True, model_name='SimpleNet' is routed to SimpleNetBN."),

    # ----- Dataset options -----
    Knob("balanced_test", "Balanced test set", "Dataset",
         widget="checkbox", dtype=bool,
         default=bool(_project_config.USE_BALANCED_TESTSET),
         source="config.py:31",
         help="Trim test set to min class size for fair per-class evaluation."),
    Knob("asl_augment", "ASL augmentation", "Dataset",
         widget="checkbox", dtype=bool, default=True,
         depends_on={"dataset_name": "ASL"},
         help="Random affine + jitter on the ASL training stream."),
    Knob("max_train_batches", "Max train batches/epoch (smoke)", "Dataset",
         widget="number", dtype=int, default=0,
         bounds=(0, 100_000, 1),
         help="0 = no cap. Caps batches per epoch for quick local runs."),

    # ----- Device characteristic source -----
    Knob("device_xlsx", "Device xlsx path", "Device",
         widget="text", dtype=str,
         default=str(_project_config.DEFAULT_DEVICE_XLSX),
         help="Path to LTP/LTD pulse-characteristic spreadsheet."),

    # ----- Energy / area accounting -----
    Knob("E_pulse_LTP_J", "Energy per LTP pulse (J)", "Energy",
         widget="number", dtype=float, default=float(_ENERGY["E_pulse_LTP_J"]),
         bounds=(0.0, 1.0, None),
         source="config.py:59",
         help="1e-12 J = 1 pJ. Placeholder — tune from datasheet."),
    Knob("E_pulse_LTD_J", "Energy per LTD pulse (J)", "Energy",
         widget="number", dtype=float, default=float(_ENERGY["E_pulse_LTD_J"]),
         bounds=(0.0, 1.0, None),
         source="config.py:60"),
    Knob("E_read_J", "Energy per read MAC (J)", "Energy",
         widget="number", dtype=float, default=float(_ENERGY["E_read_J"]),
         bounds=(0.0, 1.0, None),
         source="config.py:61",
         help="1e-15 J = 1 fJ."),
    Knob("cell_area_m2", "Cell area (m²)", "Energy",
         widget="number", dtype=float, default=float(_ENERGY["cell_area_m2"]),
         bounds=(0.0, 1.0, None),
         source="config.py:62",
         help="1e-12 m² = 1 µm²."),

    # ----- Energy-tab simulated workload (no training) -----
    Knob("energy_n_samples", "Simulated total samples (Energy tab)", "EnergyTab",
         widget="number", dtype=int, default=600_000,
         bounds=(1, 100_000_000, 1),
         help="Used by Energy tab as 'how many reads = n_samples × num_weights'."),
    Knob("energy_pulses_ltp", "Assumed total LTP pulses (Energy tab)", "EnergyTab",
         widget="number", dtype=int, default=350_000,
         bounds=(0, 100_000_000, 1)),
    Knob("energy_pulses_ltd", "Assumed total LTD pulses (Energy tab)", "EnergyTab",
         widget="number", dtype=int, default=0,
         bounds=(0, 100_000_000, 1)),

    # ----- Degradation tab -----
    Knob("deg_normal_xlsx", "Normal device xlsx", "Degradation",
         widget="text", dtype=str,
         default=str(_project_config.DEFAULT_DEVICE_XLSX)),
    Knob("deg_degraded_xlsx", "Degraded device xlsx", "Degradation",
         widget="text", dtype=str,
         default=str(_project_config.DEFAULT_DEVICE_XLSX)),
    Knob("deg_target_layers", "Target layers", "Degradation",
         widget="text", dtype=str, default="fc1,fc2",
         help="Comma-separated layer names."),
    Knob("deg_mode", "Degradation mode", "Degradation",
         widget="dropdown", dtype=str, default="ratio_single",
         choices=DEGRADATION_MODE_CHOICES,
         help="'ratio_sweep' iterates ratios; 'single' uses one; "
              "'range' / 'indices' pick specific cells."),
    Knob("deg_ratio_list", "Ratio sweep list (%)", "Degradation",
         widget="text", dtype=str, default="0, 5, 10, 20, 35, 50, 70, 90",
         depends_on={"deg_mode": "ratio_sweep"},
         help="Comma-separated percentages."),
    Knob("deg_ratio_single", "Single ratio (%)", "Degradation",
         widget="slider", dtype=float, default=20.0,
         bounds=(0.0, 100.0, 0.5),
         depends_on={"deg_mode": "ratio_single"}),
    Knob("deg_range_start", "Range start idx", "Degradation",
         widget="number", dtype=int, default=0,
         bounds=(0, 1_000_000, 1),
         depends_on={"deg_mode": "range"}),
    Knob("deg_range_end", "Range end idx", "Degradation",
         widget="number", dtype=int, default=20_000,
         bounds=(0, 1_000_000, 1),
         depends_on={"deg_mode": "range"}),
    Knob("deg_indices", "Indices (CSV)", "Degradation",
         widget="text", dtype=str, default="",
         depends_on={"deg_mode": "indices"}),
    Knob("deg_seed", "Degradation RNG seed", "Degradation",
         widget="number", dtype=int, default=42,
         bounds=(0, 2**31 - 1, 1)),

    # ----- Sweep tab -----
    Knob("sweep_models", "Sweep: models", "Sweep",
         widget="text", dtype=str,
         default="SimpleNet, Simple_CNN, LeNet5",
         help="Comma-separated subset of model choices."),
    Knob("sweep_datasets", "Sweep: datasets", "Sweep",
         widget="text", dtype=str,
         default="MNIST",
         help="Comma-separated subset of dataset choices."),
    Knob("sweep_epochs", "Sweep: epochs per cell", "Sweep",
         widget="slider", dtype=int, default=3,
         bounds=(1, 20, 1),
         help="Kept small for sweep speed."),
]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
KNOBS_BY_KEY: dict[str, Knob] = {k.key: k for k in KNOBS}


def defaults() -> dict[str, Any]:
    """Return a dict of {key: default_value} for all knobs."""
    return {k.key: k.default for k in KNOBS}


def categories() -> list[str]:
    """Distinct categories in declared order."""
    seen, out = set(), []
    for k in KNOBS:
        if k.category not in seen:
            seen.add(k.category)
            out.append(k.category)
    return out


def knobs_in(category: str) -> list[Knob]:
    return [k for k in KNOBS if k.category == category]
