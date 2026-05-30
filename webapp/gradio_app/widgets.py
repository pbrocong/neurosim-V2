"""Build Gradio widgets from the schema.

Public surface:
  - build_widgets(keys)  -> {key: gr.Component}
  - wire_dependencies(widgets) -> attaches change-handlers that flip
    `interactive=` on dependent widgets so the UI clearly shows which knobs
    are currently active.
"""
from __future__ import annotations
import gradio as gr

from .schema import KNOBS, KNOBS_BY_KEY, Knob


# ---------------------------------------------------------------------------
# One widget per Knob
# ---------------------------------------------------------------------------
def _make_one(k: Knob) -> gr.components.Component:
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
                         minimum=lo, maximum=hi, precision=(0 if k.dtype is int else None))
    if k.widget == "text":
        return gr.Textbox(value=str(k.default), label=k.label, info=info)
    raise ValueError(f"Unknown widget type: {k.widget!r}")


def build_widgets(keys: list[str]) -> dict:
    """Create widgets for the given knob keys, in order."""
    out = {}
    for key in keys:
        if key not in KNOBS_BY_KEY:
            raise KeyError(f"Unknown knob key: {key!r}")
        out[key] = _make_one(KNOBS_BY_KEY[key])
    return out


# ---------------------------------------------------------------------------
# Dependency wiring: when a parent's value changes, recompute interactive=
# for every dependent.
# ---------------------------------------------------------------------------
def wire_dependencies(widgets: dict):
    """For each knob with depends_on={parent: expected}, register a change
    handler on the parent that flips `interactive` on the dependent.

    Skips wiring for any (parent, dependent) where either side isn't present
    in the supplied widget dict (e.g., tab that only uses a subset).
    """
    # Group dependents by parent so a single parent.change call updates all
    parents: dict[str, list[tuple[str, object]]] = {}
    for k in KNOBS:
        if not k.depends_on:
            continue
        if k.key not in widgets:
            continue
        for parent_key, expected in k.depends_on.items():
            if parent_key not in widgets:
                continue
            parents.setdefault(parent_key, []).append((k.key, expected))

    for parent_key, deps in parents.items():
        dep_widgets = [widgets[dk] for dk, _ in deps]
        expected_vals = [exp for _, exp in deps]

        def _handler(parent_val, _expected_vals=expected_vals):
            return [gr.update(interactive=(parent_val == ev)) for ev in _expected_vals]

        widgets[parent_key].change(
            fn=_handler,
            inputs=[widgets[parent_key]],
            outputs=dep_widgets,
        )

        # Initial state: evaluate against the parent's default
        parent_default = KNOBS_BY_KEY[parent_key].default
        for dep_key, expected in deps:
            widgets[dep_key].interactive = (parent_default == expected)


# ---------------------------------------------------------------------------
# Grouping by category, inside collapsible accordions
# ---------------------------------------------------------------------------
def render_panel(category_keys_map: dict[str, list[str]]) -> dict:
    """Render an Accordion per category and return {key: widget} flat dict.

    `category_keys_map` is {category_label: [knob_key, ...]}; the order of
    keys in the dict drives UI order.
    """
    widgets: dict = {}
    for category, keys in category_keys_map.items():
        with gr.Accordion(category, open=(category in {"Model", "Dataset", "Training"})):
            for key in keys:
                w = _make_one(KNOBS_BY_KEY[key])
                widgets[key] = w
    return widgets


def values_from_inputs(component_keys: list[str], values: list) -> dict:
    """Pack a Gradio handler's `*inputs` tuple back into a {key: value} dict."""
    return dict(zip(component_keys, values))
