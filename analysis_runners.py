# analysis_runners.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import config

from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer, estimate_nl_from_norm_a
from data_loader import read_split_by_write_vh
from train_eval import train, test


def _safe_get_attr(obj, names, default=None):
    """여러 후보 attribute 중 존재하는 첫 번째를 반환."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def run_analysis(file_paths,
                 mode,
                 train_loader,
                 test_loader,
                 model_factory,
                 loss_mode,
                 dataset_label="",
                 v3_modes: dict | None = None):
    """
    mode:
      - "A"  : A값(정확히는 fitter가 제공하는 A 관련 값) vs Accuracy
      - "NL" : estimate_nl_from_norm_a()로 변환한 NL vs Accuracy

    model_factory: 매 파일마다 새 모델을 만들기 위한 callable (no-arg).
    loss_mode: "nll" (log_softmax 출력) 또는 "ce" (logits 출력).
    v3_modes: optional dict with keys pair_mode/pair_strategy/sigma_d2d/online
              propagated from main.py's V3 mode prompt.
    """
    if v3_modes is None:
        v3_modes = {}
    pair_mode = bool(v3_modes.get("pair_mode", False))
    pair_strategy = str(v3_modes.get("pair_strategy", "mixed"))
    sigma_d2d = float(v3_modes.get("sigma_d2d", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for f_path in file_paths:
        print("\n" + "=" * 70)
        print(f"[INFO] Processing: {f_path}")

        ltp_df, ltd_df = read_split_by_write_vh(f_path)
        if ltp_df is None or ltd_df is None:
            print("[WARNING] 데이터 로딩 실패 -> skip")
            continue

        fitter = NeuroSimFitter(
            ltp_df,
            ltd_df,
            target_range=config.TARGET_RANGE,
            ltp_fit_ratio=config.LTP_FIT_RATIO,
            ltd_fit_ratio=config.LTD_FIT_RATIO,
            use_p_start_offset=True,
            sigma_d2d=sigma_d2d,
        )

        A_ltp = _safe_get_attr(fitter, ["A_LTP_Norm", "A_LTP_Raw", "A_LTP"], default=None)
        A_ltd = _safe_get_attr(fitter, ["A_LTD_Norm", "A_LTD_Raw", "A_LTD"], default=None)

        if A_ltp is None or A_ltd is None:
            print("[WARNING] fitter에서 A 파라미터를 찾을 수 없음 -> metric=0으로 처리")
            metric_val = 0.0
        else:
            if mode == "A":
                metric_val = float((A_ltp + A_ltd) / 2.0)
            elif mode == "NL":
                nl_ltp = estimate_nl_from_norm_a(float(A_ltp))
                nl_ltd = estimate_nl_from_norm_a(float(A_ltd))
                metric_val = float((nl_ltp + nl_ltd) / 2.0)

                if hasattr(fitter, "plot_normalized_fitting"):
                    try:
                        fitter.plot_normalized_fitting(os.path.basename(f_path), nl_ltp, nl_ltd)
                    except Exception as e:
                        print(f"[WARNING] plot_normalized_fitting 실패: {e}")
            else:
                raise ValueError("mode는 'A' 또는 'NL'만 가능합니다.")

        model = model_factory().to(device)
        optimizer = NeuroSimOptimizer(
            model.parameters(),
            lr=config.LEARNING_RATE,
            fitter=fitter,
            pulse_scaling_factor=config.PULSE_SCALING_FACTOR,
            use_c2c_noise=getattr(config, "USE_C2C_NOISE", True),
            use_discretisation=getattr(config, "USE_CONDUCTANCE_DISCRETISATION", False),
            pair_mode=pair_mode,
            pair_strategy=pair_strategy,
            energy_params=getattr(config, "ENERGY_PARAMS", None),
            d2d_seed=getattr(config, "D2D_SEED", 0),
        )

        epoch_accs = []
        for ep in range(1, config.EPOCHS + 1):
            train(model, device, train_loader, optimizer, ep, loss_mode=loss_mode)
            acc = test(model, device, test_loader, loss_mode=loss_mode)
            epoch_accs.append(acc)

        final_acc = float(epoch_accs[-1])
        energy = optimizer.energy_report()
        print(f"[RESULT] Metric({mode})={metric_val:.4f}, Final Acc={final_acc:.2f}%, "
              f"E={energy['total_energy_J']*1e6:.3f}µJ, pulses={energy['total_pulses']:.0f}")

        results.append((os.path.basename(f_path), metric_val, final_acc,
                        energy["total_pulses"], energy["total_energy_J"] * 1e6))

    if not results:
        print("[INFO] 분석 결과가 없습니다.")
        return

    names = [r[0] for r in results]
    vals  = [r[1] for r in results]
    accs  = [r[2] for r in results]

    title_suffix = f" ({dataset_label})" if dataset_label else ""
    plt.figure(figsize=(10, 6))
    plt.scatter(vals, accs, s=100)
    for i, txt in enumerate(names):
        plt.annotate(txt, (vals[i], accs[i]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel(f"Average {mode} Value")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{mode} vs Accuracy{title_suffix}")
    plt.grid(True)
    plt.show()

    df = pd.DataFrame(results, columns=["File", mode, "Accuracy", "TotalPulses", "Energy_uJ"])
    out_xlsx = f"{mode}_vs_Accuracy_Result.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"[INFO] 결과 저장 완료: {out_xlsx}")