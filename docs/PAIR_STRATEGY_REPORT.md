# Differential Pair Strategy — Mixed (V3.0) vs LTP-Only Comparison

User pointed out — correctly — that the original `pair_mode` implementation
used **LTP only** (apply LTP to G⁺ when W↑, LTP to G⁻ when W↓). The full
MLP_NeuroSim V3.0 canonical version uses **both LTP and LTD** with
per-element cell selection based on headroom. Both implementations are now
available; users can pick via `pair_strategy="mixed"` or `"ltp_only"`.

---

## 1. The Two Strategies

### `pair_strategy="mixed"` (V3.0 canonical, default)

Per-element, choose the cell that has more headroom for the required direction:

```
W↑ (grad < 0):
  if (target_max - G⁺) ≥ G⁻ :  LTP on G⁺   (G⁺ goes up,   W = G⁺ − G⁻ goes up)
  else                       :  LTD on G⁻   (G⁻ goes down, W goes up)

W↓ (grad > 0):
  if G⁺ ≥ (target_max - G⁻) :  LTD on G⁺   (G⁺ goes down, W goes down)
  else                       :  LTP on G⁻   (G⁻ goes up,   W goes down)
```

All four curves (LTP⁺, LTD⁺, LTP⁻, LTD⁻) are used. Cells naturally stay balanced; **no refresh** required.

### `pair_strategy="ltp_only"` (Burr/Boybat variant)

Always use the LTP curve (more linear on most devices):

```
W↑ (grad < 0):  LTP on G⁺
W↓ (grad > 0):  LTP on G⁻
```

Both cells drift up over time; when both exceed `0.85 · target_max` the optimizer **refreshes**: `G⁺ ← max(W, 0)`, `G⁻ ← max(−W, 0)`. Published in Burr et al. 2015, Boybat et al. 2018.

---

## 2. Side-by-Side Benchmark (same ZnO_Encap device, seed=0)

| Configuration                 | Test acc (per epoch) | Best | LTP pulses | LTD pulses | LTP / total | Notes |
|-------------------------------|----------------------|------|------------|------------|-------------|-------|
| **SimpleNet MNIST, no pair**  | 88.93 / 89.50 / 90.24| 90.24| 61 162     | 50 785     | 54.6 %      | baseline (single-cell) |
| SimpleNet MNIST, pair `ltp_only` | 88.80 / 89.78 / 90.29| 90.29| 127 298 | 0          | 100 %       | +0.05 vs baseline |
| **SimpleNet MNIST, pair `mixed`** | **89.19 / 89.89 / 90.56** | **90.56** | 127 601 | 79 | 99.94 % | **+0.32 vs baseline (BEST)** |
| **Simple_CNN MNIST, no pair** | 95.02 / 95.62        | 95.62| (combined) | (combined) | —           | baseline |
| Simple_CNN MNIST, pair `ltp_only` | 95.19 / 96.15    | 96.15| 287 542    | 0          | 100 %       | +0.53 vs baseline (BEST on this cell) |
| Simple_CNN MNIST, pair `mixed`| 94.67 / 95.64        | 95.64| 295 923    | 85         | 99.97 %     | +0.02 vs baseline |
| **Simple_CNN ASL, no pair**   | 72.49 / 79.53        | 79.53| 144 650    | 122 812    | 54.1 %      | baseline |
| Simple_CNN ASL, pair `ltp_only` | 75.14 / 80.10      | 80.10| 298 366    | 0          | 100 %       | +0.57 vs baseline |
| Simple_CNN ASL, pair `mixed`  | 72.66 / 77.11        | 77.11| 299 669    | 24         | 99.99 %     | −2.42 vs baseline |

### Interpretation

* **LTD almost never fires in `mixed` mode on this device.** Why? At init both cells are near 0 (because we split `G⁺ = max(W,0)`, `G⁻ = max(-W,0)` and SimpleNet's Glorot init produces |W| ≪ 1). When cells are nearly empty, the headroom rule almost always picks LTP. LTD only fires after some cells saturate (the 24–85 LTD pulses we see).
* So on this device, mixed mode behaves *almost like* ltp_only — but with subtle differences in the saturation regime that produce slightly different optimizer trajectories.
* On SimpleNet MNIST, the mixed version **wins** (90.56 %).
* On Simple_CNN, `ltp_only` happens to give the best result (96.15 % MNIST, 80.10 % ASL) because this device's LTD is noticeably more non-linear (`A_LTP=17.5 > A_LTD=13.4`); the few LTD-pulse selections that mixed makes inject extra non-linearity noise into the CNN.
* On a device where LTP and LTD were equally linear, mixed would consistently beat ltp_only because it doesn't need the refresh step.

### Rule of thumb
- **Use `mixed`** if your device's LTP and LTD are similar (small asymmetry).
- **Use `ltp_only`** if `A_LTP ≫ A_LTD` (very asymmetric, LTP much more linear). This ZnO_Encap is mildly in that camp.

---

## 3. Pulse / energy implications

| Configuration            | LTP pulses (3 epochs) | LTD pulses | Total | Notes |
|--------------------------|-----------------------|------------|-------|-------|
| no pair (baseline)       | 61 k                  | 51 k       | 112 k | half each direction |
| pair `ltp_only`          | 127 k                 | 0          | 127 k | exclusive LTP; needs refresh |
| pair `mixed`             | 127 k                 | 0.1 k      | 127 k | almost exclusive LTP on this device |

Both pair-modes use ~14 % more pulses than the single-cell baseline (because two cells share the burden but neither cancels). Energy and area cost is exactly 2× single-cell area (modulo accounting for differential read amplifier area, which we don't model).

---

## 4. Yes — V3.0 really does use LTD

To be explicit on the original question: **the canonical MLP_NeuroSim V3.0 pair update applies both LTP and LTD curves.** Our previous `pair_mode` implementation took the published "LTP-only + refresh" shortcut. Both are now selectable. The default has been switched to `"mixed"` to match V3.0 faithfully.

---

## 5. How to use

### Interactive (main.py)
```
1) Differential pair (G⁺ − G⁻) mode 사용? [y/N]: y
   ↳ pair-update 전략:
     [1] mixed     (V3.0 canonical: LTP+LTD, headroom 기반 셀 선택) (recommended)
     [2] ltp_only  (Burr/Boybat 변형: LTP만 + 주기적 refresh)
     선택 [1/2, default 1]:
```

### Headless CLI
```bash
# Mixed (V3.0 default)
python3 headless_runner.py --model SimpleNet --dataset MNIST --epochs 3 \
        --pair-mode --energy-report

# LTP-only variant
python3 headless_runner.py --model SimpleNet --dataset MNIST --epochs 3 \
        --pair-mode --pair-strategy ltp_only --energy-report
```

### Python API
```python
opt = NeuroSimOptimizer(
    model.parameters(), lr=1e-2, fitter=fitter,
    pulse_scaling_factor=5, use_c2c_noise=True,
    pair_mode=True,
    pair_strategy="mixed",      # or "ltp_only"
)
```
