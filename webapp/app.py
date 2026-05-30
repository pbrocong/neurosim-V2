"""NeuroSim V2 — Gradio web UI.

Single deployable entry point. Run from the repo root:  `python webapp/app.py`.
On Hugging Face Spaces this file is auto-run by the runtime (see DEPLOY.md).

All hyperparameters are exposed; categorical accordions group them.
JSON export/import lets researchers persist and share configurations.
"""
from __future__ import annotations
import os
import sys

# Make the core simulator modules (config, models, train_eval, ...) importable.
# They live in  <repo>/simulator/ ; this file lives in  <repo>/webapp/ .
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "simulator"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

import gradio as gr

from gradio_app.schema import (
    KNOBS, KNOBS_BY_KEY, knobs_in, defaults,
    HEAVY_MODELS, MODEL_CHOICES, DATASET_CHOICES,
)
from gradio_app import runners, config_io


# ---------------------------------------------------------------------------
# Which knobs go on which tab
# ---------------------------------------------------------------------------
# Common knobs every tab shares (model + dataset + training + optim + device + ...).
COMMON_CATEGORIES = [
    "Model", "Dataset", "Training", "Optimizer", "Fitter",
    "Device", "Pair", "Online", "Energy",
]
QUICK_EXTRA: list[str] = []
SWEEP_EXTRA = [k.key for k in knobs_in("Sweep")]
DEGRADATION_EXTRA = [k.key for k in knobs_in("Degradation")]
ENERGY_EXTRA = [k.key for k in knobs_in("EnergyTab")]


def _common_keys() -> list[str]:
    keys: list[str] = []
    for c in COMMON_CATEGORIES:
        keys.extend(k.key for k in knobs_in(c))
    return keys


COMMON_KEYS = _common_keys()


# ---------------------------------------------------------------------------
# Widget builder used inside each tab
# ---------------------------------------------------------------------------
def _build_panel(extra_keys: list[str]) -> dict:
    """Render category accordions + (optional) extras under their own
    accordion. Returns flat {key: widget}."""
    widgets: dict = {}
    for cat in COMMON_CATEGORIES:
        with gr.Accordion(cat, open=(cat in {"Model", "Dataset", "Training"})):
            for k in knobs_in(cat):
                widgets[k.key] = _component_for(k)
    if extra_keys:
        with gr.Accordion("Tab-specific", open=True):
            for key in extra_keys:
                k = KNOBS_BY_KEY[key]
                widgets[key] = _component_for(k)
    return widgets


def _component_for(k):
    info = (k.help or "") + (f"  ({k.source})" if k.source else "")
    if k.widget == "checkbox":
        return gr.Checkbox(value=bool(k.default), label=k.label, info=info)
    if k.widget == "dropdown":
        return gr.Dropdown(choices=k.choices, value=k.default, label=k.label, info=info)
    if k.widget == "slider":
        lo, hi, step = (k.bounds or (0, 1, None))
        return gr.Slider(minimum=lo, maximum=hi, value=k.default,
                         step=step, label=k.label, info=info)
    if k.widget == "number":
        lo, hi = (k.bounds[0], k.bounds[1]) if k.bounds else (None, None)
        return gr.Number(value=k.default, label=k.label, info=info,
                         minimum=lo, maximum=hi,
                         precision=(0 if k.dtype is int else None))
    if k.widget == "text":
        return gr.Textbox(value=str(k.default), label=k.label, info=info)
    raise ValueError(f"unknown widget {k.widget!r}")


def _wire_deps_local(widgets: dict):
    """Same logic as widgets.wire_dependencies, inlined to avoid circular
    Gradio-context issues (must run inside the Blocks scope)."""
    parents: dict[str, list[tuple[str, object]]] = {}
    for k in KNOBS:
        if not k.depends_on or k.key not in widgets:
            continue
        for parent_key, expected in k.depends_on.items():
            if parent_key not in widgets:
                continue
            parents.setdefault(parent_key, []).append((k.key, expected))

    for parent_key, deps in parents.items():
        dep_widgets = [widgets[dk] for dk, _ in deps]
        expected_vals = [exp for _, exp in deps]

        def _handler(parent_val, _expected=expected_vals):
            return [gr.update(interactive=(parent_val == ev)) for ev in _expected]

        widgets[parent_key].change(_handler, [widgets[parent_key]], dep_widgets)


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------
def _make_quick_handler(widget_keys: list[str]):
    def handler(*values, progress=gr.Progress(track_tqdm=False)):
        knobs = dict(zip(widget_keys, values))
        warning_md = _heavy_model_warning(knobs["model_name"])

        def cb(ep, total, msg):
            progress(ep / max(total, 1), desc=msg)

        try:
            out = runners.run_quick(knobs, progress_cb=cb)
        except Exception as e:
            return ("❌ " + str(e),
                    None, None, None,
                    "")
        summary = (f"**Device**: {out['device']}  |  "
                   f"**Final train**: {out['final_train_acc']:.2f}%  |  "
                   f"**Final test**: {out['final_test_acc']:.2f}%")
        if warning_md:
            summary = warning_md + "\n\n" + summary
        return (
            summary,
            out["figures"]["accuracy"],
            out["figures"]["confusion"],
            {"fitter": out["fitter"], "energy_report": out["energy_report"]},
            "",
        )
    return handler


def _make_sweep_handler(widget_keys: list[str]):
    def handler(*values, progress=gr.Progress(track_tqdm=False)):
        knobs = dict(zip(widget_keys, values))

        def cb(i, total, msg):
            progress(i / max(total, 1), desc=msg)

        try:
            out = runners.run_sweep(knobs, progress_cb=cb)
        except Exception as e:
            return f"❌ {e}", None, None
        rows = out["rows"]
        # Make a list-of-lists for gr.Dataframe
        headers = list(rows[0].keys()) if rows else []
        table = [[r.get(h, "") for h in headers] for r in rows]
        return "", out["figure"], gr.update(value=table, headers=headers)
    return handler


def _make_energy_handler(widget_keys: list[str]):
    def handler(*values):
        knobs = dict(zip(widget_keys, values))
        try:
            out = runners.compute_energy(knobs)
        except Exception as e:
            return f"❌ {e}", None, None
        fig = out.pop("figure")
        return "", fig, out
    return handler


def _make_degradation_handler(widget_keys: list[str]):
    def handler(*values, progress=gr.Progress(track_tqdm=False)):
        knobs = dict(zip(widget_keys, values))

        def cb(ep, total, msg):
            progress(ep / max(total, 1), desc=msg)

        try:
            out = runners.run_degradation(knobs, progress_cb=cb)
        except Exception as e:
            return f"❌ {e}", None, None
        summary = (f"**Mode**: {out['mode']}  |  "
                   f"**Target layers**: {', '.join(out['target_layers'])}  |  "
                   f"**Final train**: {out['final_train_acc']:.2f}%  |  "
                   f"**Final test**: {out['final_test_acc']:.2f}%")
        return summary, out["figures"]["accuracy"], out["mask_stats"]
    return handler


def _heavy_model_warning(model_name: str) -> str:
    if model_name in HEAVY_MODELS:
        return ("⚠️ **Heavy model on CPU**: training may exceed HF Spaces' "
                f"~300s request limit. Consider 1–2 epochs only for `{model_name}`.")
    return ""


# ---------------------------------------------------------------------------
# JSON IO handlers
# ---------------------------------------------------------------------------
def _make_export_handler(widget_keys: list[str]):
    def handler(*values):
        knobs = dict(zip(widget_keys, values))
        path = config_io.export_config(knobs)
        return path
    return handler


def _make_import_handler(widget_keys: list[str]):
    def handler(file_obj):
        if file_obj is None:
            updates = [gr.update() for _ in widget_keys]
            return ["No file uploaded."] + updates
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        merged, warns = config_io.import_config(path)
        warn_text = "\n".join("• " + w for w in warns) if warns else "Imported."
        updates = [gr.update(value=merged[k]) for k in widget_keys]
        return [warn_text] + updates
    return handler


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="NeuroSim V2", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# NeuroSim V2 — In-Memory-Computing Simulator")
        gr.Markdown(
            "Full hyperparameter surface for the ZnO memristor pulse-domain "
            "training simulator. Each tab uses the common parameter panel "
            "(left); tab-specific knobs and outputs are on the right."
        )

        with gr.Tabs():
            # -------- Tab 1: Quick Run --------
            with gr.Tab("Quick Run"):
                with gr.Row():
                    with gr.Column(scale=2):
                        quick_w = _build_panel(QUICK_EXTRA)
                        _wire_deps_local(quick_w)
                    with gr.Column(scale=3):
                        quick_status = gr.Markdown()
                        quick_acc_plot = gr.Plot(label="Train / Test accuracy")
                        quick_cm_plot = gr.Plot(label="Confusion matrix")
                        quick_json = gr.JSON(label="Fitter + energy report")
                        quick_run = gr.Button("▶ Run", variant="primary")
                        quick_run.click(
                            _make_quick_handler(list(quick_w.keys())),
                            inputs=list(quick_w.values()),
                            outputs=[quick_status, quick_acc_plot, quick_cm_plot,
                                     quick_json, gr.State()],
                        )

            # -------- Tab 2: Sweep --------
            with gr.Tab("Sweep"):
                with gr.Row():
                    with gr.Column(scale=2):
                        sweep_w = _build_panel(SWEEP_EXTRA)
                        _wire_deps_local(sweep_w)
                    with gr.Column(scale=3):
                        sweep_status = gr.Markdown()
                        sweep_plot = gr.Plot(label="Best test acc per cell")
                        sweep_table = gr.Dataframe(label="Results")
                        sweep_run = gr.Button("▶ Run sweep", variant="primary")
                        sweep_run.click(
                            _make_sweep_handler(list(sweep_w.keys())),
                            inputs=list(sweep_w.values()),
                            outputs=[sweep_status, sweep_plot, sweep_table],
                        )

            # -------- Tab 3: Degradation --------
            with gr.Tab("Degradation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        deg_w = _build_panel(DEGRADATION_EXTRA)
                        _wire_deps_local(deg_w)
                    with gr.Column(scale=3):
                        deg_status = gr.Markdown()
                        deg_acc_plot = gr.Plot(label="Train / test accuracy")
                        deg_stats = gr.JSON(label="Per-layer degradation counts")
                        deg_run = gr.Button("▶ Run", variant="primary")
                        deg_run.click(
                            _make_degradation_handler(list(deg_w.keys())),
                            inputs=list(deg_w.values()),
                            outputs=[deg_status, deg_acc_plot, deg_stats],
                        )

            # -------- Tab 4: Energy --------
            with gr.Tab("Energy"):
                with gr.Row():
                    with gr.Column(scale=2):
                        energy_w = _build_panel(ENERGY_EXTRA)
                        _wire_deps_local(energy_w)
                    with gr.Column(scale=3):
                        energy_status = gr.Markdown()
                        energy_plot = gr.Plot(label="Write vs Read energy")
                        energy_json = gr.JSON(label="Report")
                        energy_run = gr.Button("▶ Compute", variant="primary")
                        energy_run.click(
                            _make_energy_handler(list(energy_w.keys())),
                            inputs=list(energy_w.values()),
                            outputs=[energy_status, energy_plot, energy_json],
                        )

            # -------- Tab 5: Config IO --------
            with gr.Tab("Config IO"):
                gr.Markdown("Export the **current Quick-Run tab** config to "
                            "JSON, or import a JSON to overwrite Quick-Run "
                            "widgets in bulk. Other tabs are unaffected.")
                with gr.Row():
                    export_btn = gr.Button("Export Quick-Run config")
                    export_file = gr.File(label="Downloaded JSON", interactive=False)
                with gr.Row():
                    import_file = gr.File(label="Upload JSON to apply", file_types=[".json"])
                    import_btn = gr.Button("Apply import")
                import_log = gr.Markdown()

                quick_keys = list(quick_w.keys())
                export_btn.click(_make_export_handler(quick_keys),
                                 inputs=list(quick_w.values()),
                                 outputs=[export_file])
                import_btn.click(_make_import_handler(quick_keys),
                                 inputs=[import_file],
                                 outputs=[import_log] + list(quick_w.values()))

        gr.Markdown(
            "_Defaults pulled from `config.py`. Knob source files annotated "
            "in each widget's tooltip._"
        )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue(default_concurrency_limit=1).launch(show_api=False)
