import os
import time
import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # 엑셀 생성을 위해 추가됨
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st

import config
from data_loader import read_split_by_write_vh, get_mnist_loaders, get_asl_loaders
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer
from models import SimpleNet, LeNet5_MNIST
from train_eval import train, train_online, test

# Default paths: prefer the V2-local device xlsx; fall back to the legacy
# absolute paths for users who still have the original layout on disk.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_V2_XLSX = getattr(config, "DEFAULT_DEVICE_XLSX",
                           os.path.join(_ROOT, "datasets", "zno_encap_48h",
                                        "0627_Pulse characteristic#1.xlsx"))
_LEGACY_N = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/정상소자.xlsx"
_LEGACY_D = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/열화소자.xlsx"

DEFAULT_NORMAL_XLSX = _LEGACY_N if os.path.exists(_LEGACY_N) else _DEFAULT_V2_XLSX
DEFAULT_DEGRADED_XLSX = _LEGACY_D if os.path.exists(_LEGACY_D) else _DEFAULT_V2_XLSX


# ---------------------------------------------------------------------------
# 엑셀 변환 유틸리티 (추가됨)
# ---------------------------------------------------------------------------
def get_excel_download_bytes(results: List[Dict]) -> bytes:
    """결과 리스트를 바탕으로 DataFrame을 만들고 엑셀 바이너리 데이터를 반환합니다."""
    data = []
    for r in results:
        row_data = {
            "Model": r.get("model_name", "Unknown"),
            "Target Layers": "+".join(r["target_layers"]),
            "Mode": r["mode"],
            "Degraded Ratio (%)": round(r["deg_pct"], 2),
            "Degraded Cells": r["num_degraded"],
            "Total Target Cells": r["total_target_cells"],
            "Range Start": r["range"][0],
            "Range End": r["range"][1],
            "Indices": ", ".join(str(i) for i in r.get("indices", [])),
            "Final Acc (%)": round(r["final_acc"], 2),
            # V3.0 mode metadata (present from V3-enabled runs onward)
            "Pair Mode": r.get("pair_mode", False),
            "Pair Strategy": r.get("pair_strategy", "-"),
            "Online Training": r.get("use_online", False),
            "LTP Pulses": int(r.get("pulses_ltp", 0)),
            "LTD Pulses": int(r.get("pulses_ltd", 0)),
            "Energy (µJ)": r.get("energy_uJ", 0),
            "Area (µm²)": r.get("area_um2", 0),
        }
        # Epoch 별 정확도를 열(Column)로 추가
        for i, acc in enumerate(r["acc_hist"], start=1):
            row_data[f"Epoch {i}"] = round(acc, 2)
        data.append(row_data)

    df = pd.DataFrame(data)
    buffer = io.BytesIO()
    # openpyxl 엔진을 사용하여 엑셀 파일 생성
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Epoch_Accuracy')
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="정상/열화 소자 LTP/LTD 피팅 중...")
def load_fitters(normal_path: str, degraded_path: str,
                 use_offset: bool, force_zero: bool,
                 sigma_d2d_normal: float = 0.0,
                 sigma_d2d_degraded: float = 0.0):
    n_ltp, n_ltd = read_split_by_write_vh(normal_path)
    d_ltp, d_ltd = read_split_by_write_vh(degraded_path)
    if n_ltp is None or d_ltp is None:
        raise FileNotFoundError("LTP/LTD 엑셀 로드 실패")

    normal = NeuroSimFitter(
        n_ltp, n_ltd,
        target_range=config.TARGET_RANGE,
        ltp_fit_ratio=config.LTP_FIT_RATIO,
        ltd_fit_ratio=config.LTD_FIT_RATIO,
        use_p_start_offset=use_offset,
        p_start_force_zero=force_zero,
        sigma_d2d=float(sigma_d2d_normal),
    )
    degraded = NeuroSimFitter(
        d_ltp, d_ltd,
        target_range=config.TARGET_RANGE,
        ltp_fit_ratio=config.LTP_FIT_RATIO,
        ltd_fit_ratio=config.LTD_FIT_RATIO,
        use_p_start_offset=use_offset,
        p_start_force_zero=force_zero,
        sigma_d2d=float(sigma_d2d_degraded),
    )
    return normal, degraded, (n_ltp, n_ltd), (d_ltp, d_ltd)


@st.cache_resource(show_spinner="MNIST 로더 준비 중...")
def cached_mnist_loaders(batch_size: int, balanced_test: bool):
    return get_mnist_loaders(batch_size, balanced_test=balanced_test)


@st.cache_resource(show_spinner="ASL 로더 준비 중...")
def cached_asl_loaders(batch_size: int):
    return get_asl_loaders(batch_size)


# ---------------------------------------------------------------------------
# 모델/마스크
# ---------------------------------------------------------------------------
def build_model(kind: str, device):
    w_min, w_max = config.TARGET_RANGE
    if kind == "MNIST · SimpleNet (fc1, fc2)":
        return SimpleNet(w_min, w_max, use_bn=config.USE_BATCHNORM).to(device), "SimpleNet"
    elif kind == "ASL · LeNet5 (fc1, fc2, fc3)":
        return LeNet5_MNIST(w_min, w_max, num_classes=24).to(device), "LeNet5"
    raise ValueError(kind)


def list_target_layer_options(kind: str) -> List[str]:
    if kind.startswith("MNIST"):
        return ["fc1", "fc2"]
    return ["fc1", "fc2", "fc3"]


def build_layer_masks(model: nn.Module,
                      target_layer_names: List[str],
                      mode: str,
                      ratio: float,
                      start_idx: int,
                      end_idx: int,
                      device,
                      seed: int = 42,
                      indices: List[int] = None,
                      ) -> Tuple[Dict, int, int, List[Tuple]]:
    weight_entries = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_entries.append((name, m.weight))

    masks = {
        id(p): torch.ones_like(p.data, dtype=torch.bool, device=device)
        for _, p in weight_entries
    }

    target_entries = [(n, p) for n, p in weight_entries if n in target_layer_names]
    if not target_entries:
        return masks, 0, 0, []

    total_target = int(sum(p.numel() for _, p in target_entries))

    flat_mask = torch.ones(total_target, dtype=torch.bool, device=device)
    if mode == "ratio":
        n_deg = int(round(total_target * float(ratio)))
        n_deg = max(0, min(total_target, n_deg))
        if n_deg > 0:
            rng = np.random.default_rng(seed)
            deg_positions = rng.choice(total_target, size=n_deg, replace=False)
            flat_mask[torch.from_numpy(deg_positions).long()] = False
    elif mode == "range":
        s = max(0, int(start_idx))
        e = min(total_target, int(end_idx))
        if s < e:
            flat_mask[s:e] = False
        n_deg = int((~flat_mask).sum().item())
    elif mode == "indices":
        # 사용자가 지정한 1D 인덱스 위치만 열화 처리
        idx_list = indices or []
        valid = sorted({int(i) for i in idx_list if 0 <= int(i) < total_target})
        if valid:
            flat_mask[torch.tensor(valid, dtype=torch.long, device=device)] = False
        n_deg = int((~flat_mask).sum().item())
    else:
        raise ValueError(mode)

    per_layer = []
    offset = 0
    for name, p in target_entries:
        n = p.numel()
        m = flat_mask[offset:offset + n].reshape(p.shape).clone()
        masks[id(p)] = m
        per_layer.append((name, tuple(p.shape), n, int((~m).sum().item())))
        offset += n

    return masks, total_target, int((~flat_mask).sum().item()), per_layer


# ---------------------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------------------
def run_one(model_kind: str,
            target_layer_names: List[str],
            mode: str,
            ratio: float,
            start_idx: int,
            end_idx: int,
            epochs: int,
            lr: float,
            psf: float,
            batch_size: int,
            seed: int,
            normal_fitter: NeuroSimFitter,
            degraded_fitter: NeuroSimFitter,
            indices: List[int] = None,
            # V3.0 advanced modes
            pair_mode: bool = False,
            pair_strategy: str = "mixed",
            use_online: bool = False,
            online_micro_batch: int = 1,
            online_max_samples: int = None,
            log_cb=None) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_kind.startswith("MNIST"):
        train_loader, test_loader = cached_mnist_loaders(batch_size, config.USE_BALANCED_TESTSET)
        loss_mode = "nll"
    else:
        train_loader, test_loader, _ = cached_asl_loaders(batch_size)
        loss_mode = "ce"

    model, model_name = build_model(model_kind, device)
    masks, total_target, n_deg, per_layer = build_layer_masks(
        model, target_layer_names, mode, ratio, start_idx, end_idx, device,
        seed=seed, indices=indices,
    )

    optimizer = NeuroSimOptimizer(
        model.parameters(), lr, normal_fitter, psf,
        degraded_fitter=degraded_fitter, masks=masks,
        pair_mode=bool(pair_mode),
        pair_strategy=str(pair_strategy),
        energy_params=getattr(config, "ENERGY_PARAMS", None),
        d2d_seed=int(seed),
    )

    if log_cb:
        extra = []
        if pair_mode: extra.append(f"pair={pair_strategy}")
        if use_online: extra.append(f"online({online_micro_batch})")
        extra_s = ("  [" + ",".join(extra) + "]") if extra else ""
        log_cb(f"[{mode}] target={target_layer_names}  "
               f"deg={n_deg}/{total_target} "
               f"({(n_deg / total_target * 100 if total_target else 0):.2f}%)  "
               f"model={model_name}{extra_s}")

    acc_hist = []
    for epoch in range(1, epochs + 1):
        if use_online:
            train_online(model, device, train_loader, optimizer, epoch,
                         loss_mode=loss_mode,
                         micro_batch=int(online_micro_batch),
                         max_samples_per_epoch=online_max_samples,
                         progress_every=0)
        else:
            train(model, device, train_loader, optimizer, epoch, loss_mode=loss_mode)
        acc = test(model, device, test_loader, loss_mode=loss_mode)
        acc_hist.append(acc)
        if log_cb:
            log_cb(f"  epoch {epoch}/{epochs}  acc={acc:.2f}%")

    energy = optimizer.energy_report()
    return {
        "model_name": model_name,
        "target_layers": list(target_layer_names),
        "mode": mode,
        "ratio": float(ratio),
        "range": (int(start_idx), int(end_idx)),
        "indices": list(indices) if indices else [],
        "total_target_cells": total_target,
        "num_degraded": n_deg,
        "deg_pct": (n_deg / total_target * 100 if total_target else 0.0),
        "acc_hist": acc_hist,
        "final_acc": float(acc_hist[-1]) if acc_hist else float("nan"),
        "per_layer": per_layer,
        # V3 modes record
        "pair_mode": bool(pair_mode),
        "pair_strategy": str(pair_strategy) if pair_mode else "-",
        "use_online": bool(use_online),
        "energy_uJ": round(energy["total_energy_J"] * 1e6, 4),
        "pulses_ltp": float(energy["total_pulses_ltp"]),
        "pulses_ltd": float(energy["total_pulses_ltd"]),
        "area_um2": round(energy["array_area_m2"] * 1e12, 1),
    }


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_epoch_curves(results: List[Dict], title: str = "Epoch-wise Test Accuracy"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in results:
        ep = range(1, len(r["acc_hist"]) + 1)
        label = (f"{'+'.join(r['target_layers'])}  "
                 f"{r['deg_pct']:.1f}% deg  ({r['num_degraded']}/{r['total_target_cells']})")
        ax.plot(ep, r["acc_hist"], marker="o", label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_ratio_vs_acc(results: List[Dict], title: str = "Degradation Ratio vs Final Accuracy"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    by_layers: Dict[str, List[Dict]] = {}
    for r in results:
        key = "+".join(r["target_layers"])
        by_layers.setdefault(key, []).append(r)

    for key, lst in by_layers.items():
        lst = sorted(lst, key=lambda x: x["deg_pct"])
        xs = [r["deg_pct"] for r in lst]
        ys = [r["final_acc"] for r in lst]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"target = {key}")

    ax.set_xlabel("Degradation Ratio (%)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_fitter_preview(normal_fitter: NeuroSimFitter,
                        degraded_fitter: NeuroSimFitter,
                        ltp_dfs, ltd_dfs):
    n_ltp_df, n_ltd_df = ltp_dfs[0], ltd_dfs[0]
    d_ltp_df, d_ltd_df = ltp_dfs[1], ltd_dfs[1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for fitter, ltp_df, ltd_df, color, tag in [
        (normal_fitter, n_ltp_df, n_ltd_df, "tab:blue", "Normal"),
        (degraded_fitter, d_ltp_df, d_ltd_df, "tab:red", "Degraded"),
    ]:
        ltp_p = ltp_df["PulseNum"].values.astype(np.float32)
        ltd_p = ltd_df["PulseNum"].values.astype(np.float32)
        ltp_g = ltp_df["Conductance"].values.astype(np.float32)
        ltd_g = ltd_df["Conductance"].values.astype(np.float32)

        ltp_n = fitter._normalize_0_1_real(ltp_g)
        ltd_n = fitter._normalize_0_1_real(ltd_g)

        with torch.no_grad():
            ltp_fit = fitter.unscale(fitter.g_of_p_ltp(torch.from_numpy(ltp_p)).cpu().numpy())
            ltd_fit = fitter.unscale(fitter.g_of_p_ltd(torch.from_numpy(ltd_p)).cpu().numpy())
        ltp_fit_n = fitter._normalize_0_1_real(ltp_fit)
        ltd_fit_n = fitter._normalize_0_1_real(ltd_fit)

        # positive pulses (LTP) on the right, negative pulses (LTD) on the left
        ltp_x = ltp_p - ltp_p.min()
        ltd_x = -(ltd_p - ltd_p.min())
        ax.scatter(ltp_x, ltp_n, s=10, alpha=0.5, color=color)
        ax.scatter(ltd_x, ltd_n, s=10, alpha=0.5, color=color)
        ax.plot(np.sort(ltp_x), ltp_fit_n[np.argsort(ltp_x)], color=color, linewidth=2,
                label=f"{tag} LTP fit")
        ax.plot(np.sort(ltd_x), ltd_fit_n[np.argsort(ltd_x)], color=color, linewidth=2,
                linestyle="--", label=f"{tag} LTD fit")

    ax.axvline(0.0, color="gray", lw=1.0, ls="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("← negative pulses (LTD)   |   positive pulses (LTP) →")
    ax.set_ylabel("Normalized Conductance (0~1)")
    ax.set_title("LTP / LTD Fit Preview (Normal vs Degraded)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="망가진 소자 시뮬레이션", layout="wide")
    st.title("망가진 소자(RRAM) 시뮬레이션")
    st.caption("정상/열화 소자 LTP-LTD 엑셀 → SimpleNet/LeNet5 학습 → 정확도 비교")

    # --- 파일 경로 입력 + 검증 ---
    with st.expander("입력 파일 경로", expanded=True):
        normal_xlsx_raw = st.text_input(
            "정상 소자 엑셀 경로",
            value=st.session_state.get("normal_xlsx", DEFAULT_NORMAL_XLSX),
            key="normal_xlsx",
        )
        degraded_xlsx_raw = st.text_input(
            "열화 소자 엑셀 경로",
            value=st.session_state.get("degraded_xlsx", DEFAULT_DEGRADED_XLSX),
            key="degraded_xlsx",
        )
        normal_xlsx = os.path.expanduser(normal_xlsx_raw.strip().strip("'").strip('"'))
        degraded_xlsx = os.path.expanduser(degraded_xlsx_raw.strip().strip("'").strip('"'))
        if not os.path.exists(normal_xlsx):
            st.error(f"정상 소자 엑셀이 없습니다: {normal_xlsx}")
            st.stop()
        if not os.path.exists(degraded_xlsx):
            st.error(f"열화 소자 엑셀이 없습니다: {degraded_xlsx}")
            st.stop()

    # --- 사이드바 (실행 전 파라미터) ---
    with st.sidebar:
        st.header("시뮬레이션 설정")

        model_kind = st.selectbox(
            "데이터셋 / 모델",
            ["MNIST · SimpleNet (fc1, fc2)", "ASL · LeNet5 (fc1, fc2, fc3)"],
            index=0,
        )
        layer_options = list_target_layer_options(model_kind)

        st.subheader("적용 레이어")
        layer_mode = st.radio(
            "열화 적용 범위",
            ["fc1 only", "fc1 + fc2", "fc1 only / fc1+fc2 비교"],
            index=2,
        )

        st.subheader("열화 모드")
        mode = st.radio(
            "모드",
            ["비율(%) sweep", "비율(%) 단일", "위치(start, end)", "특정 인덱스"],
            index=0,
        )

        ratios_str = ""
        ratio_single = None
        start_idx = end_idx = 0
        indices_input: List[int] = []

        if mode == "비율(%) sweep":
            ratios_str = st.text_input(
                "비율 리스트 (콤마, 0~100)",
                value="0, 5, 10, 20, 35, 50, 70, 90",
            )
        elif mode == "비율(%) 단일":
            ratio_single = st.slider("열화 비율(%)", 0.0, 100.0, 20.0, 1.0)
        elif mode == "위치(start, end)":
            start_idx = st.number_input("start_idx", min_value=0, value=0, step=1)
            end_idx = st.number_input("end_idx", min_value=0, value=20000, step=100)
        else:  # "특정 인덱스"
            n_idx = int(st.number_input(
                "입력할 좌표 개수",
                min_value=1, max_value=1_000_000, value=3, step=1,
                help="모델의 전체 target cells 수까지 가능. "
                     "개별 입력 모드는 위젯 수가 많아지면 느려지므로 ~200개 이하 권장.",
            ))
            input_style = st.radio(
                "입력 방식",
                ["개별 입력", "콤마/공백 구분 텍스트"],
                index=(0 if n_idx <= 200 else 1),
                horizontal=True,
            )
            if input_style == "개별 입력":
                if n_idx > 200:
                    st.warning(
                        f"개별 입력 모드는 위젯 {n_idx}개를 생성하므로 매우 느려질 수 있습니다. "
                        "200개 초과 시 '콤마/공백 구분 텍스트' 모드를 권장합니다."
                    )
                cols_per_row = 4
                for row in range((n_idx + cols_per_row - 1) // cols_per_row):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        i = row * cols_per_row + c
                        if i < n_idx:
                            with cols[c]:
                                v = st.number_input(
                                    f"#{i+1}",
                                    min_value=0,
                                    value=int(st.session_state.get(f"idx_{i}", i)),
                                    step=1,
                                    key=f"idx_{i}",
                                )
                                indices_input.append(int(v))
            else:
                raw = st.text_area(
                    f"{n_idx}개의 인덱스 (콤마/공백/개행 구분)",
                    value=", ".join(str(i) for i in range(n_idx)),
                    height=100,
                )
                tokens = [t for t in raw.replace(",", " ").split() if t.strip()]
                try:
                    parsed = [int(t) for t in tokens]
                except ValueError:
                    st.error("인덱스 파싱 실패: 정수만 입력하세요.")
                    parsed = []
                if len(parsed) != n_idx:
                    st.warning(f"입력 개수({len(parsed)}) ≠ 지정 개수({n_idx}). "
                               f"앞에서 {n_idx}개만 사용합니다.")
                indices_input = parsed[:n_idx]

            if indices_input:
                st.caption(f"적용 인덱스: {sorted(set(indices_input))}")

        st.subheader("학습 하이퍼파라미터")
        epochs = st.slider("Epochs", 1, 30, value=int(config.EPOCHS))
        lr = st.number_input("Learning Rate", value=float(config.LEARNING_RATE),
                             format="%.6f", step=1e-5)
        psf = st.number_input("Pulse Scaling Factor", value=float(config.PULSE_SCALING_FACTOR), step=10.0)
        batch_size = st.number_input("Batch size", value=int(config.BATCH_SIZE), step=32, min_value=16)
        seed = st.number_input("Seed", value=42, step=1)

        st.subheader("Fitter 옵션")
        use_offset = st.checkbox("use_p_start_offset (API compat, no-op)", value=True)
        force_zero = st.checkbox("p_start_force_zero (API compat, no-op)", value=True)

        st.subheader("V3.0 advanced modes")
        v3_pair = st.checkbox(
            "Differential pair (G⁺ − G⁻)",
            value=bool(getattr(config, "USE_PAIR_MODE", False)),
            help="셀당 2개의 conductance 셀로 weight 표현. 면적 2배.",
        )
        v3_pair_strategy = "mixed"
        if v3_pair:
            v3_pair_strategy = st.radio(
                "Pair update 전략",
                ["mixed (V3.0 canonical, LTP+LTD)", "ltp_only (Burr/Boybat 변형)"],
                index=0,
                horizontal=False,
            ).split()[0]
        v3_d2d_n = st.number_input(
            "σ_d2d (정상 fitter)",
            value=float(getattr(config, "SIGMA_D2D", 0.0)),
            min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
            help="0 = OFF. 셀별 A 변동(log-normal). 정상 셀에 적용.",
        )
        v3_d2d_d = st.number_input(
            "σ_d2d (열화 fitter)",
            value=float(getattr(config, "SIGMA_D2D", 0.0)),
            min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
            help="0 = OFF. 열화 셀의 셀별 A 변동.",
        )
        v3_online = st.checkbox(
            "Online (batch=1) training mode",
            value=False,
            help="셀당 펄스 1회씩. BatchNorm 자동으로 eval 모드로.",
        )
        v3_online_mb = 1
        v3_online_cap = None
        if v3_online:
            v3_online_mb = int(st.number_input(
                "online micro_batch", min_value=1, max_value=64, value=1, step=1))
            v3_online_cap_input = int(st.number_input(
                "online max_samples_per_epoch (0=무제한)", min_value=0, value=0, step=1000))
            v3_online_cap = None if v3_online_cap_input == 0 else v3_online_cap_input

        run_btn = st.button("시뮬레이션 실행", type="primary", use_container_width=True)
        clear_btn = st.button("이전 결과 초기화", use_container_width=True)

    if clear_btn:
        st.session_state.pop("results", None)
        st.session_state.pop("last_meta", None)
        st.rerun()

    # --- Fitter 미리보기 ---
    try:
        normal_fitter, degraded_fitter, n_dfs, d_dfs = load_fitters(
            normal_xlsx, degraded_xlsx, use_offset, force_zero,
            sigma_d2d_normal=float(v3_d2d_n),
            sigma_d2d_degraded=float(v3_d2d_d),
        )
    except Exception as e:
        st.error(f"Fitter 로드 실패: {e}")
        st.stop()

    with st.expander("LTP/LTD 피팅 곡선 비교 (정상 vs 열화)", expanded=False):
        st.pyplot(plot_fitter_preview(normal_fitter, degraded_fitter,
                                      (n_dfs[0], d_dfs[0]), (n_dfs[1], d_dfs[1])))

    # --- 실행 ---
    if run_btn:
        if layer_mode == "fc1 only":
            layer_sets = [["fc1"]]
        elif layer_mode == "fc1 + fc2":
            layer_sets = [["fc1", "fc2"]]
        else:
            layer_sets = [["fc1"], ["fc1", "fc2"]]

        if mode == "비율(%) sweep":
            try:
                ratios_pct = [float(x) for x in ratios_str.replace(" ", "").split(",") if x]
            except Exception:
                st.error("비율 리스트 파싱 실패. 예: 0, 5, 20, 50")
                st.stop()
            run_mode = "ratio"
            ratios = [r / 100.0 for r in ratios_pct]
            ranges = [(0, 0)] * len(ratios)
            indices_per_run = [None] * len(ratios)
        elif mode == "비율(%) 단일":
            run_mode = "ratio"
            ratios = [ratio_single / 100.0]
            ranges = [(0, 0)]
            indices_per_run = [None]
        elif mode == "위치(start, end)":
            run_mode = "range"
            ratios = [0.0]
            ranges = [(int(start_idx), int(end_idx))]
            indices_per_run = [None]
        else:  # "특정 인덱스"
            if not indices_input:
                st.error("적용할 인덱스가 비어있습니다.")
                st.stop()
            run_mode = "indices"
            ratios = [0.0]
            ranges = [(0, 0)]
            indices_per_run = [list(indices_input)]

        total_runs = len(layer_sets) * len(ratios)
        st.info(f"총 {total_runs}개 시나리오 실행 (epoch {epochs}회씩)")
        progress = st.progress(0.0)
        log_area = st.empty()
        log_buf = []

        def log(msg):
            log_buf.append(msg)
            log_area.code("\n".join(log_buf[-30:]), language="text")

        results: List[Dict] = []
        run_idx = 0
        t0 = time.time()
        for tlayers in layer_sets:
            for r, (s, e), idxs in zip(ratios, ranges, indices_per_run):
                run_idx += 1
                idx_str = (f"  indices={idxs[:8]}{'...' if idxs and len(idxs) > 8 else ''}"
                           if idxs else "")
                log(f"\n=== [{run_idx}/{total_runs}] target={tlayers}  "
                    f"mode={run_mode}  ratio={r:.3f}  range=({s},{e}){idx_str} ===")
                res = run_one(
                    model_kind=model_kind,
                    target_layer_names=tlayers,
                    mode=run_mode,
                    ratio=r,
                    start_idx=s,
                    end_idx=e,
                    epochs=epochs,
                    lr=lr,
                    psf=psf,
                    batch_size=batch_size,
                    seed=seed,
                    normal_fitter=normal_fitter,
                    degraded_fitter=degraded_fitter,
                    indices=idxs,
                    pair_mode=bool(v3_pair),
                    pair_strategy=str(v3_pair_strategy),
                    use_online=bool(v3_online),
                    online_micro_batch=int(v3_online_mb),
                    online_max_samples=v3_online_cap,
                    log_cb=log,
                )
                results.append(res)
                progress.progress(run_idx / total_runs)

        elapsed = time.time() - t0
        log(f"\n전체 소요 시간: {elapsed:.1f}s")

        st.session_state["results"] = results
        st.session_state["last_meta"] = {
            "mode": mode, "model_kind": model_kind, "epochs": epochs,
        }

    # --- 결과 표시 ---
    if "results" in st.session_state:
        results = st.session_state["results"]
        meta = st.session_state.get("last_meta", {})

        # --- 데이터 내보내기 버튼 (추가된 부분) ---
        st.subheader("데이터 내보내기")
        try:
            excel_bytes = get_excel_download_bytes(results)
            st.download_button(
                label="📥 에포크 별 정확도 엑셀 파일 다운로드 (.xlsx)",
                data=excel_bytes,
                file_name=f"epoch_accuracy_results_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
        except Exception as e:
            st.error(f"엑셀 파일 생성 중 오류가 발생했습니다: {e} (openpyxl 패키지 설치 여부를 확인해주세요.)")
            
        st.markdown("---")
        st.subheader("결과 요약")
        rows = []
        for r in results:
            idxs = r.get("indices", [])
            idx_disp = (
                f"{idxs[:6]}{'...' if len(idxs) > 6 else ''}" if idxs else "-"
            )
            rows.append({
                "target": "+".join(r["target_layers"]),
                "mode": r["mode"],
                "ratio(%)": f"{r['deg_pct']:.2f}",
                "deg/total": f"{r['num_degraded']}/{r['total_target_cells']}",
                "range": f"{r['range'][0]}~{r['range'][1]}",
                "indices": idx_disp,
                "final acc(%)": f"{r['final_acc']:.2f}",
                # V3 mode columns (back-compat: only present if those fields exist)
                "pair": r.get("pair_strategy", "-") if r.get("pair_mode", False) else "-",
                "online": "yes" if r.get("use_online", False) else "-",
                "energy(µJ)": f"{r.get('energy_uJ', 0):.3f}",
                "LTP_p": int(r.get("pulses_ltp", 0)),
                "LTD_p": int(r.get("pulses_ltd", 0)),
            })
        st.dataframe(rows, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Epoch-wise 정확도**")
            st.pyplot(plot_epoch_curves(results,
                      title=f"Epoch Acc ({meta.get('model_kind','')})"))
        with col2:
            st.markdown("**열화 비율 vs 최종 정확도**")
            st.pyplot(plot_ratio_vs_acc(results,
                      title=f"Degradation Ratio vs Final Acc ({meta.get('model_kind','')})"))

        with st.expander("레이어별 열화 셀 분포 (마지막 시나리오 기준)"):
            last = results[-1]
            st.json({
                "target_layers": last["target_layers"],
                "per_layer": [
                    {"name": n, "shape": list(s), "n": n_, "degraded": d}
                    for (n, s, n_, d) in last["per_layer"]
                ],
                "summary": {
                    "total_target_cells": last["total_target_cells"],
                    "num_degraded": last["num_degraded"],
                    "deg_pct": last["deg_pct"],
                },
            })


if __name__ == "__main__":
    main()