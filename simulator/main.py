# main.py
from __future__ import annotations
import os
import sys
import subprocess
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import config
from data_loader import (
    read_split_by_write_vh,
    get_loaders,
    get_asl_loaders,
    get_class_names,
    DATASET_REGISTRY,
)
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer
from models import (SimpleNet, Simple_CNN, Standard_CNN, ResNet18_MNIST,
                    VGG_MNIST, AlexNet_MNIST, LeNet5_MNIST)
from train_eval import train, train_online, test, test_specific_letters
from analysis_runners import run_analysis


def clean_path(path_str):
    return path_str.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# 모델 레지스트리
#   value = (factory(w_min, w_max, num_classes) -> nn.Module, loss_mode)
#   * SimpleNet 은 log_softmax 출력 -> nll
#   * 그 외 모델은 logits 출력 -> ce
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "SimpleNet (MLP)":
        (lambda wmin, wmax, nc: SimpleNet(wmin, wmax, use_bn=config.USE_BATCHNORM, num_classes=nc), "nll"),
    "Simple CNN (Conv 1개)":
        (lambda wmin, wmax, nc: Simple_CNN(wmin, wmax, num_classes=nc), "ce"),
    "Standard CNN (Conv 2개)":
        (lambda wmin, wmax, nc: Standard_CNN(wmin, wmax, num_classes=nc), "ce"),
    "LeNet-5":
        (lambda wmin, wmax, nc: LeNet5_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "VGG-like":
        (lambda wmin, wmax, nc: VGG_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "AlexNet-like (불안정)":
        (lambda wmin, wmax, nc: AlexNet_MNIST(wmin, wmax, num_classes=nc), "ce"),
    "ResNet18 (변형)":
        (lambda wmin, wmax, nc: ResNet18_MNIST(wmin, wmax, num_classes=nc), "ce"),
}


# ---------------------------------------------------------------------------
# 메뉴 헬퍼
# ---------------------------------------------------------------------------
def _prompt_choice(title, options):
    """options 리스트 중 하나를 고르게 하고 인덱스(0-based) 반환."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")
    while True:
        try:
            choice = int(input(f"선택 (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                return choice - 1
        except ValueError:
            pass
        print(f"1에서 {len(options)} 사이의 숫자를 입력해주세요.")


def select_dataset():
    options = list(DATASET_REGISTRY.keys()) + ["ASL (Sign Language MNIST)"]
    idx = _prompt_choice("데이터셋을 선택하세요", options)
    return options[idx]


def select_model():
    keys = list(MODEL_REGISTRY.keys())
    idx = _prompt_choice("모델을 선택하세요 (models.py)", keys)
    return keys[idx]


def select_analysis():
    options = [
        "학습 + 정확도 (단일 엑셀)",
        "A 값 vs Accuracy 분석 (다중 엑셀)",
        "NL 값 vs Accuracy 분석 (다중 엑셀)",
        "망가진 소자 시뮬레이션 (Streamlit UI)",
    ]
    idx = _prompt_choice("분석 모드를 선택하세요", options)
    return idx  # 0..3


# ---------------------------------------------------------------------------
# Advanced V3.0 mode selection (pair / σ_d2d / online / energy)
# ---------------------------------------------------------------------------
def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            ans = input(prompt + suffix).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return default
        if not ans:
            return default
        if ans in ("y", "yes", "예", "ㅇ"):  return True
        if ans in ("n", "no", "아니오", "ㄴ"):  return False
        print("y 또는 n 으로 답해주세요.")


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        try:
            ans = input(f"{prompt} (default {default}): ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not ans:
            return default
        try:
            return float(ans)
        except ValueError:
            print("숫자를 입력해주세요.")


def select_v3_modes():
    """터미널에서 V3.0 advanced mode 들을 켤지/끌지 묻기."""
    print("\n" + "=" * 50)
    print("V3.0 advanced modes (기본값 = 전부 OFF, 이전 결과 그대로)")
    print("=" * 50)
    pair = _prompt_yes_no("1) Differential pair (G⁺ − G⁻) mode 사용?",
                          default=bool(getattr(config, "USE_PAIR_MODE", False)))
    pair_strategy = getattr(config, "PAIR_STRATEGY", "mixed")
    if pair:
        print("   ↳ pair-update 전략:")
        print("     [1] mixed     (V3.0 canonical: LTP+LTD, headroom 기반 셀 선택) (recommended)")
        print("     [2] ltp_only  (Burr/Boybat 변형: LTP만 + 주기적 refresh)")
        try:
            sel = input("     선택 [1/2, default 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            sel = ""
        pair_strategy = "ltp_only" if sel == "2" else "mixed"
    use_d2d = _prompt_yes_no("2) Device-to-device variation (σ_d2d) 사용?",
                             default=bool(getattr(config, "SIGMA_D2D", 0.0) > 0))
    sigma_d2d = (_prompt_float("   ↳ σ_d2d 값",
                               default=getattr(config, "SIGMA_D2D", 0.10) or 0.10)
                 if use_d2d else 0.0)
    online = _prompt_yes_no("3) Online (batch=1) training mode 사용?",
                            default=bool(getattr(config, "USE_ONLINE_TRAINING", False)))
    energy = _prompt_yes_no("4) Pulse-energy 회계 결과 출력?", default=True)
    return {
        "pair_mode":     pair,
        "pair_strategy": pair_strategy,
        "sigma_d2d":     float(sigma_d2d),
        "online":        online,
        "energy":        energy,
    }


# ---------------------------------------------------------------------------
# 데이터셋 로딩 (ASL 포함)
# ---------------------------------------------------------------------------
def load_dataset(dataset_name):
    """
    return: (train_loader, test_loader, num_classes, asl_test_dataset_or_None)
    asl_test_dataset_or_None 은 ASL 일 때만 GACHON 평가용으로 사용.
    """
    if dataset_name.startswith("ASL"):
        train_loader, test_loader, test_dataset = get_asl_loaders(config.BATCH_SIZE)
        if train_loader is None:
            return None, None, None, None
        return train_loader, test_loader, 24, test_dataset

    train_loader, test_loader, num_classes = get_loaders(
        dataset_name, config.BATCH_SIZE,
        balanced_test=config.USE_BALANCED_TESTSET,
    )
    return train_loader, test_loader, num_classes, None


# ---------------------------------------------------------------------------
# 분석 실행기
# ---------------------------------------------------------------------------
def plot_confusion_matrix(targets, preds, class_names, title):
    """numpy 기반 confusion matrix 시각화 (sklearn 의존 X)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1

    # 행 합으로 정규화한 정확도 행렬(시각용)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    size = max(6.0, min(2.0 + n * 0.45, 16.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap='Blues', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Row-normalized')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    rotation = 0 if all(len(c) <= 2 for c in class_names) else 45
    ax.set_xticklabels(class_names, rotation=rotation, ha='right' if rotation else 'center', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title)

    # 클래스 수가 적당히 적을 때만 셀 값 표시
    if n <= 20:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=8)
    fig.tight_layout()
    return fig


def run_training(dataset_name, model_key, num_classes, train_loader, test_loader,
                 asl_test_dataset, file_path, v3_modes: dict | None = None):
    """단일 엑셀 파일로 학습 + 정확도 그래프 + Confusion Matrix.

    v3_modes : dict | None
        반환된 V3.0 모드 옵션. None 이면 config 기본값 사용.
        keys: pair_mode (bool), sigma_d2d (float), online (bool), energy (bool)
    """
    if v3_modes is None:
        v3_modes = {
            "pair_mode":     getattr(config, "USE_PAIR_MODE", False),
            "pair_strategy": getattr(config, "PAIR_STRATEGY", "mixed"),
            "sigma_d2d":     getattr(config, "SIGMA_D2D", 0.0),
            "online":        getattr(config, "USE_ONLINE_TRAINING", False),
            "energy":        True,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ltp, ltd = read_split_by_write_vh(clean_path(file_path))
    if ltp is None:
        return

    # ASL 만 offset 옵션을 config 의 ASL_* 로 분기 (기존 동작 유지)
    is_asl = dataset_name.startswith("ASL")
    fitter = NeuroSimFitter(
        ltp, ltd,
        target_range=config.TARGET_RANGE,
        ltp_fit_ratio=config.LTP_FIT_RATIO,
        ltd_fit_ratio=config.LTD_FIT_RATIO,
        use_p_start_offset=(config.ASL_USE_P_START_OFFSET if is_asl else True),
        p_start_force_zero=(config.ASL_P_START_FORCE_ZERO if is_asl else False),
        sigma_d2d=float(v3_modes.get("sigma_d2d", 0.0) or 0.0),
    )
    fitter.plot_fit_normalized_0_1(ltp, ltd, title_suffix=f"({dataset_name})")

    factory, loss_mode = MODEL_REGISTRY[model_key]
    w_min, w_max = config.TARGET_RANGE
    model = factory(w_min, w_max, num_classes).to(device)
    optimizer = NeuroSimOptimizer(
        model.parameters(), config.LEARNING_RATE, fitter, config.PULSE_SCALING_FACTOR,
        use_c2c_noise=getattr(config, "USE_C2C_NOISE", True),
        use_discretisation=getattr(config, "USE_CONDUCTANCE_DISCRETISATION", False),
        pair_mode=bool(v3_modes.get("pair_mode", False)),
        pair_strategy=str(v3_modes.get("pair_strategy", "mixed")),
        energy_params=getattr(config, "ENERGY_PARAMS", None),
        d2d_seed=getattr(config, "D2D_SEED", 0),
    )

    use_scheduler = is_asl
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) if use_scheduler else None

    print(f"[INFO] 학습 시작  dataset={dataset_name}  model={model_key}  loss={loss_mode}")
    train_acc_hist = []
    test_acc_hist = []
    final_preds = None
    final_targets = None
    gachon_hist = {c: [] for c in "GACHON"} if (is_asl and asl_test_dataset is not None) else None

    online = bool(v3_modes.get("online", False))
    if online:
        print("[INFO] ⚡ Online (batch=1) 학습 모드 활성 — BatchNorm 은 eval 모드로 전환됨")

    for epoch in range(1, config.EPOCHS + 1):
        if online:
            tr_acc = train_online(
                model, device, train_loader, optimizer, epoch,
                loss_mode=loss_mode,
                micro_batch=getattr(config, "ONLINE_MICRO_BATCH", 1),
                max_samples_per_epoch=getattr(config, "ONLINE_MAX_SAMPLES_PER_EPOCH", None),
            )
        else:
            tr_acc = train(model, device, train_loader, optimizer, epoch, loss_mode=loss_mode)
        train_acc_hist.append(tr_acc)
        if scheduler is not None:
            scheduler.step()
            print(f"  >> Current LR: {scheduler.get_last_lr()[0]:.6f}")

        # 마지막 epoch 에서만 confusion matrix 용 predictions 수집
        if epoch == config.EPOCHS:
            te_acc, final_preds, final_targets = test(
                model, device, test_loader, loss_mode=loss_mode, return_preds=True
            )
        else:
            te_acc = test(model, device, test_loader, loss_mode=loss_mode)
        test_acc_hist.append(te_acc)
        print(f"Epoch {epoch}  train={tr_acc:.2f}%  test={te_acc:.2f}%")

        if gachon_hist is not None:
            spec_acc = test_specific_letters(model, device, asl_test_dataset, "GACHON")
            for c, v in spec_acc.items():
                gachon_hist[c].append(v)
            print("-" * 50)

    # --- 1) Train vs Test Accuracy 곡선 ---
    epochs_range = list(range(1, config.EPOCHS + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_acc_hist, marker='o', color='tab:blue', label='Train')
    plt.plot(epochs_range, test_acc_hist, marker='s', color='tab:red', label='Test')
    plt.title(f'Train vs Test Accuracy ({dataset_name} / {model_key})')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 2) Confusion Matrix (test set, 마지막 epoch) ---
    if final_preds is not None and final_targets is not None:
        class_names = get_class_names(dataset_name, num_classes)
        # ASL 의 경우 class_names 길이가 24와 일치하는지 안전 체크
        if len(class_names) != num_classes:
            class_names = [str(i) for i in range(num_classes)]
        cm_fig = plot_confusion_matrix(
            final_targets, final_preds, class_names,
            title=f'Confusion Matrix ({dataset_name} / {model_key}) - final epoch',
        )
        plt.show()

    # --- 3) (ASL 전용) GACHON 글자별 정확도 ---
    if gachon_hist is not None:
        plt.figure(figsize=(12, 8))
        for c, v in gachon_hist.items():
            plt.plot(epochs_range, v, marker='o', label=f'Accuracy of "{c}"')
        plt.title('Epoch-wise Accuracy for Specific Letters (GACHON)')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

    # --- 4) Pulse-energy 회계 ---
    if v3_modes.get("energy", True):
        rep = optimizer.energy_report()
        print("\n" + "=" * 50)
        print("⚡ Pulse-Energy / Area Report")
        print("=" * 50)
        print(f"  total weights     : {rep['num_weights']:,}")
        print(f"  pair_mode         : {rep['pair_mode']}")
        print(f"  LTP pulses        : {rep['total_pulses_ltp']:.0f}")
        print(f"  LTD pulses        : {rep['total_pulses_ltd']:.0f}")
        print(f"  Total pulses      : {rep['total_pulses']:.0f}")
        print(f"  Reads             : {rep['total_reads']:,}")
        print(f"  Write energy      : {rep['write_energy_J']*1e6:.4f}  µJ")
        print(f"  Read energy       : {rep['read_energy_J']*1e6:.4f}  µJ")
        print(f"  Total energy      : {rep['total_energy_J']*1e6:.4f}  µJ")
        print(f"  Crossbar area     : {rep['array_area_m2']*1e12:.2f}  µm²")


def run_a_or_nl_analysis(mode_label, dataset_name, model_key, num_classes,
                         train_loader, test_loader, file_paths,
                         v3_modes: dict | None = None):
    """analysis_runners.run_analysis 호출 래퍼."""
    factory, loss_mode = MODEL_REGISTRY[model_key]
    w_min, w_max = config.TARGET_RANGE

    def model_factory():
        return factory(w_min, w_max, num_classes)

    run_analysis(
        file_paths,
        mode=mode_label,                # "A" or "NL"
        train_loader=train_loader,
        test_loader=test_loader,
        model_factory=model_factory,
        loss_mode=loss_mode,
        dataset_label=f"{dataset_name} / {model_key}",
        v3_modes=v3_modes,
    )


def launch_degradation_ui():
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "degradation_ui.py")
    print(f"[INFO] Streamlit UI 실행: {script_path}")
    print("[INFO] UI 내부에서 데이터셋·모델·열화 모드를 다시 선택해야 합니다.")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])


# ---------------------------------------------------------------------------
# 메인 흐름
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: dataset
    dataset_name = select_dataset()

    # Step 2: model
    model_key = select_model()

    # Step 3: analysis
    analysis_idx = select_analysis()

    print(f"\n[설정] dataset={dataset_name}  model={model_key}  "
          f"analysis={analysis_idx + 1}")

    # 망가진 소자 시뮬레이션: 데이터셋/모델 로드 없이 바로 UI
    if analysis_idx == 3:
        launch_degradation_ui()
        sys.exit(0)

    # 데이터셋 로딩 (학습 + A/NL 분석에서 공통 사용)
    train_loader, test_loader, num_classes, asl_test_dataset = load_dataset(dataset_name)
    if train_loader is None:
        print("[ERROR] 데이터셋 로딩 실패")
        sys.exit(1)

    if analysis_idx == 0:
        raw = input("엑셀 파일 경로: ")
        # Step 4: V3.0 advanced modes (pair / σ_d2d / online / energy)
        v3_modes = select_v3_modes()
        run_training(dataset_name, model_key, num_classes,
                     train_loader, test_loader, asl_test_dataset, raw,
                     v3_modes=v3_modes)

    elif analysis_idx in (1, 2):
        raw = input("엑셀 파일 경로들 (콤마 구분): ").split(',')
        paths = [clean_path(p) for p in raw if p.strip()]
        if not paths:
            print("[ERROR] 파일 경로가 비어있습니다.")
            sys.exit(1)
        mode_label = "A" if analysis_idx == 1 else "NL"
        v3_modes = select_v3_modes()
        run_a_or_nl_analysis(mode_label, dataset_name, model_key, num_classes,
                             train_loader, test_loader, paths,
                             v3_modes=v3_modes)
