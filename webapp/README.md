---
title: NeuroSim V2
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# NeuroSim V2 — In-Memory-Computing Simulator (Gradio)

A web UI for the ZnO-memristor pulse-domain training simulator. **Every
hyperparameter** the simulator supports is exposed as a widget, with
defaults pulled from `config.py` and tooltips that cite each knob's source
line.

## Tabs

- **Quick Run** — train one (model × dataset) end-to-end; outputs train/test
  accuracy curve, confusion matrix, and energy/area report.
- **Sweep** — cross-product run over selected models × datasets at small
  epoch budgets; produces a comparison table and bar chart.
- **Degradation** — port of the previous Streamlit degradation tool;
  applies degraded-device fitter to selected layers, retrains, and
  reports the resulting accuracy.
- **Energy** — pure-arithmetic energy/area calculation with no training;
  responds instantly. Lets researchers explore sensitivity to per-pulse
  / per-read / per-area assumptions.
- **Config IO** — export the current Quick-Run knob values to JSON or
  upload a JSON to apply them.

## Running locally

```bash
# from the repo root
pip install -r webapp/requirements.txt
python webapp/app.py
# open http://127.0.0.1:7860
```

## ASL dataset

The `sign_mnist_train.csv` and `sign_mnist_test.csv` files are **not**
bundled with this Space (they're ~104 MB). To use the ASL dataset
locally, place them under `datasets/`.

## CPU constraints (HF Spaces free tier)

- Free tier is 2 vCPU / 16 GB RAM and requests time out around 5 minutes.
- `SimpleNet`, `Simple_CNN`, `LeNet5` complete in time for short epoch
  counts on MNIST/Fashion-MNIST.
- `VGG`, `AlexNet`, `ResNet18` are flagged as heavy — they will likely
  time out for non-trivial sweeps. The UI warns when one is selected.

## Code map

| File | Role |
|---|---|
| `webapp/app.py` | Gradio entry point. Builds the 5-tab interface. |
| `webapp/gradio_app/schema.py` | Single source of truth for the hyperparameters. |
| `webapp/gradio_app/widgets.py` | Auto-generates widgets from the schema; wires dependency enable/disable. |
| `webapp/gradio_app/runners.py` | Thin programmatic wrappers around `train`/`test`/`NeuroSimFitter`/`NeuroSimOptimizer`. |
| `webapp/gradio_app/config_io.py` | JSON config export/import. |
| `simulator/config.py` | Original default values. |
| `simulator/{train_eval,neurosim_utils,data_loader,models}.py` | Core compute modules — unchanged by the UI layer. |

> The web app lives in `webapp/`; the simulation engine lives in `simulator/`.
> `webapp/app.py` adds `simulator/` to `sys.path` at startup, so run it from
> the repo root: `python webapp/app.py`. For Hugging Face Spaces deployment
> see **[DEPLOY.md](DEPLOY.md)**.
