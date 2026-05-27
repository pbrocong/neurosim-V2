# MLP_NeuroSim V3.0 — Research Notes & Audit Plan

## 1. Reference: MLP_NeuroSim V3.0 (Shimeng Yu group, Georgia Tech / ASU)

Authoritative algorithm comes from:
- Param.cpp / Param.h (numWeightBit, NL, σ_c2c, σ_d2d, maxNumLevelLTP/LTD)
- NeuroSim/NonlinearWeight.cpp + lookuptable.py (NL → A mapping)
- Cell.cpp::RealDevice::Read(), Write()
- Train.cpp (epoch loop, weight-to-pulse conversion)

### 1.1 NonlinearWeight model

For *LTP* (positive write pulses), starting at G_min:
```
G_LTP(P) = B_LTP · (1 − exp(−P / A_LTP)) + G_min
B_LTP = (G_max − G_min) / (1 − exp(−Pmax_LTP / A_LTP))
```

For *LTD* (negative write pulses), starting at G_max:
```
G_LTD(P) = −B_LTD · (1 − exp(−P / A_LTD)) + G_max
B_LTD = (G_max − G_min) / (1 − exp(−Pmax_LTD / A_LTD))
```

`A` is the curvature parameter (non-linearity). NL=0 → very linear (A→∞).
NL>0 means *concave-down* LTP (rapidly saturating) which is the bad case.

### 1.2 Weight update rule (Train.cpp)

For every gradient step on a synapse:
```
ΔW   = −lr · ∂L/∂W
ΔG   = ΔW · (G_max − G_min) / (W_max − W_min)
dP   = (P_target − P_current)              # in pulse count
dP_q = sign(dP) · floor(|dP| + U(0,1))     # stochastic rounding
P_new = clip(P_current + dP_q, 0, Pmax)
G_new = NonlinearWeight(P_new, LTP or LTD)
G_new += σ_c2c · N(0,1)                    # cycle-to-cycle noise
W_new = (G_new − G_min)/(G_max−G_min)·(W_max−W_min) + W_min
```

`σ_c2c` ≈ measured std of conductance after a single write pulse.

### 1.3 Pair-cell mapping (W = G⁺ − G⁻)

The reference simulator uses a **differential pair**: every weight is two
conductance cells, and W ∝ (G⁺ − G⁻). Updating a positive ΔW means
applying LTP to G⁺ *or* LTD to G⁻. This halves the dynamic range needed per
cell and naturally suppresses asymmetry.

### 1.4 Real-to-Sim gap minimization

Things that drive the gap (and that the V3.0 reference fixes):
1. **Per-device A, B fit** to measured pulse-vs-G data.
2. **maxNumLevelLTP / LTD** = exactly the number of write pulses in the
   experiment (e.g. 64 / 128 / 256), not infinity.
3. **Pulse-domain update**, never direct W := W + ΔW.
4. **Stochastic rounding** of dP to integer pulses (above).
5. **σ_c2c, σ_d2d** injection from measured variation.
6. **Conductance discretization** (numConductanceStates per cell, default = 2^bits).
7. **G_min, G_max** taken from experiment, not arbitrary.
8. **Weight clamping** at every step.

## 2. Current Code Audit — Deviations from V3.0

Findings against the user's `neurosim_utils.py`, `models.py`, `data_loader.py`:

| # | File | Issue | V3.0 expected | Impact |
|---|---|---|---|---|
| A1 | neurosim_utils.NeuroSimFitter | LTD model `-B(1-exp(-(P-p0)/A))+G_max_scaled` flips slope but `g_max_fit_scaled` is set to `scale(ltd_g_real[0])` rather than `target_max`. With a *linear* device whose first LTD point ≈ G_max, this is fine; with noisy data this introduces 5–10 % offset error. | Use exact `G_max = target_max`. | ★★ |
| A2 | NeuroSimFitter | No `maxNumLevelLTP/LTD` enforcement. Pulse state can drift beyond Pmax → curves saturate but no clamping → grad noise. | clamp P ∈ [0, Pmax]. | ★★★ |
| A3 | NeuroSimOptimizer.step | dP rounded by `floor(dP + rand)` is exactly V3 stochastic rounding ✅ | ✅ | – |
| A4 | NeuroSimOptimizer | No σ_c2c injection after each pulse. | `G += σ·N(0,1)`. | ★★ |
| A5 | NeuroSimOptimizer | Single-cell W, no differential pair. Bidirectional updates work via two pulse-state buffers but conductance saturates asymmetrically. | W = G⁺ − G⁻. | ★★ |
| A6 | NeuroSimOptimizer | When `grad==0` exactly, neither `ltp_mask` nor `ltd_mask` is true → P state never resyncs; minor. | – | ★ |
| A7 | NeuroSimOptimizer | After step `p.data.copy_(G_new.clamp(...))`, then **recomputes P from G via `p_of_g_*`**. This loses information when an LTD pulse moved us closer to G_max than the *previous* LTD baseline; specifically the pulse counter is reset using the LTP/LTD inverse, which is only consistent if the cell stays monotone. Causes oscillation when grad sign flips often. | Keep separate pulse counters per direction (already does), but do not overwrite them on the opposite direction. The recompute should only happen on the branch that actually fired. | ★★★ |
| B1 | models.SimpleNet | Init `uniform_(−1,1)` on `fc1 ∈ ℝ^{128×784}` → output std ≈ √(784/3) ≈ 16. With BN it's OK; with `USE_BATCHNORM=False` (current default) `tanh` saturates → grad ≈ 0 for the first epoch. | Scaled uniform `±√(6/fan_in) · (w_max−w_min)/2` (Glorot in the bounded range). | ★★★ |
| B2 | models.Simple_CNN, Standard_CNN, LeNet5 | Same uniform init disaster, plus Conv2d kernel 5×5×1 → output std ≈ √(25/3) ≈ 2.9 with full clipping by ReLU. | Glorot/Kaiming within `[w_min, w_max]`. | ★★★ |
| B3 | models.* | Bias initialised to 0 ✅ (good). | – | – |
| B4 | models.Standard_CNN.fc | Two Dropout layers (0.5, 0.3) on top of NeuroSim-quantized weights amplifies stochastic update noise. | drop the second Dropout for CNN+NeuroSim. | ★ |
| C1 | data_loader | `config.ASL_TRAIN_PATH` points to a missing path on user's old machine. CSVs are present *inside* the V2 folder. | Update to V2 folder paths. | ★★★ |
| C2 | data_loader.SignMNISTDataset | `pixels uint8` is converted to PIL then `ToTensor` and `Normalize((0.5,),(0.5,))` → [-1,1] range ✅. | – | – |
| C3 | data_loader.get_*_loaders | `num_workers=0` (default), `shuffle=True` for train ✅. MNIST normalization (0.1307, 0.3081) outputs unbounded → consistent with reference. | – | – |
| D1 | config | LR=1e-4 + PULSE_SCALING_FACTOR=300 effectively makes one optimizer step equivalent to ~3% of A per gradient unit. With NL≈0 linear device A is huge, so dP ≈ 0 per step → no learning. Need higher PSF for linear devices. | scale PSF to match A. | ★★★ |
| D2 | config | EPOCHS=10 too short for CNN on ASL. | 30+ for CNN. | ★★ |

The single-most-likely root cause of the CNN-on-linear-device accuracy collapse
is the **combination of B2 (broken init) + D1 (vanishing pulse step) +
A7 (cross-direction P-counter overwrite)**. Each one alone would just slow
training; together they make CNN gradients effectively zero.

## 3. Refactor Plan

1. Rewrite `neurosim_utils.py`:
   - Introduce `Pmax_LTP, Pmax_LTD` parsed from data (`int(max(PulseNum))`).
   - Add `sigma_c2c` parameter (optional, default = empirical std of residuals).
   - Add `numConductanceStates` discretization (optional).
   - Add `_init_pulse_states` that picks the *closer* of `p_of_g_ltp` /
     `p_of_g_ltd` for the starting G — currently we initialise *both* with the
     same G which is consistent, but on update we must **only touch the counter
     for the direction we actually pulsed**.
   - Add `update_sigma_c2c=…`.
   - Optional differential-pair mode `pair_mode=True`.
2. Rewrite weight init in `models.py`: glorot-uniform within `[w_min, w_max]`.
3. Fix `data_loader.py` to point to the V2-local CSVs.
4. Bump `config.EPOCHS`, parametrise PSF differently for MLP vs CNN.
5. Add `headless_runner.py` so we can iterate without GUI prompts.

## 4. Verification Plan

For each (model, dataset) pair, train with the **ZnO_Encap_Oven_48h_2** linear
device for the listed epochs and confirm:

| Model        | Dataset       | Target test acc | Source         |
|--------------|---------------|-----------------|----------------|
| SimpleNet    | MNIST         | ≥ 95 %          | Chen 2018      |
| SimpleNet    | ASL           | ≥ 85 %          | empirical      |
| LeNet5       | MNIST         | ≥ 98 %          | LeCun, Chen    |
| Simple_CNN   | MNIST         | ≥ 98 %          | empirical      |
| Simple_CNN   | ASL           | ≥ 90 %          | Kaggle SOTA    |
| Standard_CNN | ASL           | ≥ 93 %          | Kaggle SOTA    |

If any number is missed by > 3 %, loop back: tune PSF, learning rate, or init.
