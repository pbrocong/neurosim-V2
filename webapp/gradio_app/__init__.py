"""Gradio web UI for NeuroSim V2.

Exposes every simulator hyperparameter as a widget. Single-source-of-truth
metadata lives in `schema.py`; everything else (widgets, JSON IO, runners)
is driven from that schema.
"""
