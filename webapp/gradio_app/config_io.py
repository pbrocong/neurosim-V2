"""JSON config import/export, schema-validated."""
from __future__ import annotations
import json
import tempfile
import os
from typing import Any

from .schema import KNOBS, KNOBS_BY_KEY, defaults


def export_config(values: dict) -> str:
    """Serialize a knob-values dict to a JSON file and return its path.

    Gradio's `gr.File` requires a path on disk for downloads.
    """
    payload = {"neurosim_config_version": 1, "knobs": _coerce_for_json(values)}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                      encoding="utf-8", prefix="neurosim_config_")
    json.dump(payload, tmp, indent=2)
    tmp.close()
    return tmp.name


def import_config(path_or_text: str) -> tuple[dict, list[str]]:
    """Parse a JSON config (path OR raw text) into a values dict.

    Returns (merged_values, warnings). Unknown keys and dependency violations
    yield warnings; missing keys are filled with defaults.
    """
    raw = _read(path_or_text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return defaults(), [f"JSON parse error: {e}"]

    knobs_in = payload.get("knobs") if isinstance(payload, dict) else None
    if not isinstance(knobs_in, dict):
        # Allow a flat dict at top level for permissive import
        knobs_in = payload if isinstance(payload, dict) else {}

    base = defaults()
    warnings: list[str] = []
    for key, val in knobs_in.items():
        if key not in KNOBS_BY_KEY:
            warnings.append(f"unknown key ignored: {key}")
            continue
        k = KNOBS_BY_KEY[key]
        try:
            base[key] = _coerce(val, k)
        except (TypeError, ValueError) as e:
            warnings.append(f"could not coerce {key}={val!r}: {e}")

    # Soft check: only warn when user customized an inactive knob
    for k in KNOBS:
        if not k.depends_on or k.key not in knobs_in:
            continue
        if base.get(k.key) == k.default:
            continue  # value matches default → not a meaningful customization
        for parent, expected in k.depends_on.items():
            if base.get(parent) != expected:
                warnings.append(
                    f"{k.key} customized but its parent {parent} != {expected!r} "
                    f"(value will be ignored at runtime)"
                )

    return base, warnings


def _read(path_or_text: str) -> str:
    if path_or_text is None:
        return "{}"
    s = str(path_or_text)
    # Heuristic: looks like JSON if it starts with { or [
    if s.lstrip().startswith(("{", "[")):
        return s
    if os.path.exists(s):
        with open(s, encoding="utf-8") as f:
            return f.read()
    return s


def _coerce(val: Any, k) -> Any:
    if k.dtype is bool:
        if isinstance(val, str):
            return val.strip().lower() in ("true", "1", "yes", "on")
        return bool(val)
    if k.dtype is int:
        return int(val)
    if k.dtype is float:
        return float(val)
    return str(val)


def _coerce_for_json(values: dict) -> dict:
    """Strip non-JSON-serializable values."""
    out = {}
    for key, val in values.items():
        if key not in KNOBS_BY_KEY:
            continue
        try:
            json.dumps(val)
            out[key] = val
        except TypeError:
            out[key] = str(val)
    return out
