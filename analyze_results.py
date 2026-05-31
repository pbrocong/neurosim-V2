"""analyze_results.py — analyze an all_test.py results folder.

Reads the summary.csv produced by all_test.py and writes an `analysis/`
sub-folder with comparison charts and a text report. The headline analysis
is the good-device vs bad-device comparison; it also breaks accuracy down by
model, dataset, augmentation, and (if swept) the pair / online / sigma_d2d
modes.

Usage:
    python analyze_results.py [results_folder_or_summary.csv]

If no path is given, the most recent results/all_test_* folder is used.
"""
from __future__ import annotations
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# Columns that together identify one configuration apart from the device.
CONFIG_KEYS = ["model", "dataset", "augment",
               "pair_mode", "pair_strategy", "online", "sigma_d2d"]


# ---------------------------------------------------------------------------
# Locating + loading the results
# ---------------------------------------------------------------------------
def find_summary(path):
    """Resolve the user's argument to a summary.csv path + its folder."""
    if path and os.path.isfile(path):
        return path, os.path.dirname(os.path.abspath(path))
    if path and os.path.isdir(path):
        csv = os.path.join(path, "summary.csv")
        if os.path.isfile(csv):
            return csv, os.path.abspath(path)
        raise FileNotFoundError(f"no summary.csv in {path}")
    # default: latest results/all_test_* folder
    cands = sorted(glob.glob(os.path.join(_HERE, "results", "all_test_*")))
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "summary.csv"))]
    if not cands:
        raise FileNotFoundError(
            "no results folder found. Run all_test.py first, or pass a path.")
    folder = cands[-1]
    return os.path.join(folder, "summary.csv"), folder


def load(csv_path):
    df = pd.read_csv(csv_path)
    # keep only successful runs for accuracy analysis
    if "error" in df.columns:
        df["error"] = df["error"].fillna("")
        ok = df[df["error"].astype(str).str.len() == 0].copy()
    else:
        ok = df.copy()
    for col in ("final_test_acc", "final_train_acc", "best_test_acc",
                "total_energy_J", "sigma_d2d"):
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    return df, ok


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_device_scatter(ok, out):
    """good vs bad test accuracy, paired by configuration."""
    if not {"good", "bad"}.issubset(set(ok["device_label"].unique())):
        return None
    g = ok[ok.device_label == "good"].set_index(CONFIG_KEYS)["final_test_acc"]
    b = ok[ok.device_label == "bad"].set_index(CONFIG_KEYS)["final_test_acc"]
    j = pd.concat([g.rename("good"), b.rename("bad")], axis=1).dropna()
    if j.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(j["bad"], j["good"], alpha=0.6, s=20)
    lo = float(min(j["bad"].min(), j["good"].min())) - 2
    hi = float(max(j["bad"].max(), j["good"].max())) + 2
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="good = bad")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("bad device  test acc (%)")
    ax.set_ylabel("good device test acc (%)")
    ax.set_title("good vs bad device (per configuration)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return j


def fig_device_gap_heatmap(ok, out):
    """good − bad test-acc gap as a model × dataset heatmap."""
    if not {"good", "bad"}.issubset(set(ok["device_label"].unique())):
        return
    g = ok[ok.device_label == "good"].pivot_table(
        index="model", columns="dataset", values="final_test_acc", aggfunc="mean")
    b = ok[ok.device_label == "bad"].pivot_table(
        index="model", columns="dataset", values="final_test_acc", aggfunc="mean")
    gap = (g - b)
    if gap.empty:
        return
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * gap.shape[1], 1.2 + 0.7 * gap.shape[0]))
    vmax = float(np.nanmax(np.abs(gap.values))) or 1.0
    im = ax.imshow(gap.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="good − bad  test acc (%)")
    ax.set_xticks(range(gap.shape[1])); ax.set_xticklabels(gap.columns, rotation=45, ha="right")
    ax.set_yticks(range(gap.shape[0])); ax.set_yticklabels(gap.index)
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Device gap (good − bad) by model × dataset")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_acc_heatmap(ok, out):
    """mean test acc as model × dataset heatmap (all devices)."""
    piv = ok.pivot_table(index="model", columns="dataset",
                         values="final_test_acc", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * piv.shape[1], 1.2 + 0.7 * piv.shape[0]))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    fig.colorbar(im, ax=ax, label="mean test acc (%)")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if v < 55 else "black")
    ax.set_title("Mean test acc by model × dataset")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_factor_effects(ok, out):
    """One bar panel per factor: mean test acc grouped by that factor's values."""
    factors = [("device_label", "device"), ("model", "model"),
               ("dataset", "dataset"), ("augment", "augment"),
               ("pair_mode", "pair"), ("pair_strategy", "pair strategy"),
               ("online", "online"), ("sigma_d2d", "sigma_d2d")]
    # keep only factors that actually vary
    factors = [(c, t) for c, t in factors
               if c in ok.columns and ok[c].nunique() > 1]
    if not factors:
        return
    ncol = 2
    nrow = (len(factors) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, title) in zip(axes, factors):
        grp = ok.groupby(col)["final_test_acc"].mean().sort_index()
        ax.bar([str(x) for x in grp.index], grp.values, color="#3a78c2")
        for i, v in enumerate(grp.values):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"mean test acc by {title}")
        ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    for ax in axes[len(factors):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def fig_acc_vs_energy(ok, out):
    if "total_energy_J" not in ok.columns or ok["total_energy_J"].notna().sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for dev, sub in ok.groupby("device_label"):
        ax.scatter(sub["total_energy_J"] * 1e6, sub["final_test_acc"],
                   alpha=0.6, s=18, label=str(dev))
    ax.set_xlabel("total energy (µJ)"); ax.set_ylabel("test acc (%)")
    ax.set_title("accuracy vs energy"); ax.set_ylim(0, 105)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def write_report(df, ok, paired, out_md):
    L = ["# all_test 결과 분석", ""]
    L.append(f"- 전체 runs: **{len(df)}**  (성공 {len(ok)}, 에러 {len(df) - len(ok)})")
    if not ok.empty:
        L.append(f"- 평균 test acc: **{ok['final_test_acc'].mean():.2f}%**  "
                 f"(최고 {ok['final_test_acc'].max():.2f}%, 최저 {ok['final_test_acc'].min():.2f}%)")
    L.append("")

    # device headline
    if {"good", "bad"}.issubset(set(ok["device_label"].unique())):
        gm = ok[ok.device_label == "good"]["final_test_acc"].mean()
        bm = ok[ok.device_label == "bad"]["final_test_acc"].mean()
        L += ["## 소자 비교 (good vs bad)", "",
              f"- good 평균 test acc: **{gm:.2f}%**",
              f"- bad  평균 test acc: **{bm:.2f}%**",
              f"- 평균 차이 (good − bad): **{gm - bm:+.2f}%p**", ""]
        if paired is not None and not paired.empty:
            d = (paired["good"] - paired["bad"])
            worst = d.sort_values().head(5)
            L.append("- 나쁜 소자에서 가장 크게 떨어진 설정 (good−bad):")
            for idx, val in worst.items():
                L.append(f"    - {dict(zip(CONFIG_KEYS, idx))} → {val:+.2f}%p")
            L.append("")

    # factor effects
    L.append("## 요인별 평균 test acc")
    for col, title in [("model", "모델"), ("dataset", "데이터셋"),
                       ("augment", "augmentation"), ("pair_mode", "pair"),
                       ("pair_strategy", "pair strategy"), ("online", "online"),
                       ("sigma_d2d", "sigma_d2d")]:
        if col in ok.columns and ok[col].nunique() > 1:
            grp = ok.groupby(col)["final_test_acc"].mean().sort_values(ascending=False)
            L.append(f"- **{title}**: " +
                     ", ".join(f"{k}={v:.1f}%" for k, v in grp.items()))
    L.append("")

    # best / worst configs
    cols = [c for c in ["device_label", "model", "dataset", "augment",
                        "pair_mode", "pair_strategy", "online", "sigma_d2d",
                        "final_test_acc"] if c in ok.columns]
    top = ok.sort_values("final_test_acc", ascending=False).head(10)[cols]
    bot = ok.sort_values("final_test_acc").head(10)[cols]
    L += ["## 최고 정확도 10개", "", top.to_string(index=False), "",
          "## 최저 정확도 10개", "", bot.to_string(index=False), ""]

    # errors
    errs = df[df.get("error", "").astype(str).str.len() > 0] if "error" in df.columns else df.iloc[0:0]
    if len(errs):
        L += [f"## 에러 {len(errs)}건", ""]
        for _, r in errs.head(30).iterrows():
            L.append(f"- {r.get('device_label','')}/{r.get('model','')}/"
                     f"{r.get('dataset','')} aug={r.get('augment','')}: "
                     f"{str(r.get('error',''))[:80]}")
        L.append("")

    text = "\n".join(L)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
def analyze(path=None):
    """Analyze a results folder/CSV. Returns the analysis-folder path (or None).

    Importable so all_test.py can call it automatically when a sweep finishes.
    """
    csv_path, folder = find_summary(path)
    print(f"분석 대상: {csv_path}")
    df, ok = load(csv_path)
    if ok.empty:
        print("성공한 run 이 없습니다 (전부 에러). summary.csv 의 error 열을 확인하세요.")
        return None

    out = os.path.join(folder, "analysis")
    os.makedirs(out, exist_ok=True)

    paired = fig_device_scatter(ok, os.path.join(out, "device_good_vs_bad.png"))
    fig_device_gap_heatmap(ok, os.path.join(out, "device_gap_heatmap.png"))
    fig_acc_heatmap(ok, os.path.join(out, "acc_heatmap_model_dataset.png"))
    fig_factor_effects(ok, os.path.join(out, "factor_effects.png"))
    fig_acc_vs_energy(ok, os.path.join(out, "acc_vs_energy.png"))
    report = write_report(df, ok, paired, os.path.join(out, "analysis_report.md"))

    print(f"\n분석 결과 폴더: {out}")
    print("  - device_good_vs_bad.png      (소자 비교 산점도)")
    print("  - device_gap_heatmap.png      (good−bad 격차 히트맵)")
    print("  - acc_heatmap_model_dataset.png")
    print("  - factor_effects.png          (요인별 평균 정확도)")
    print("  - acc_vs_energy.png")
    print("  - analysis_report.md          (텍스트 요약)")
    print("\n" + "=" * 60)
    print(report[:1500])
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(0 if analyze(arg) else 1)
