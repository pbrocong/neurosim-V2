# Complete Change Log — Original Code → Current State

이 문서는 최초의 V2 코드부터 현재까지의 **모든 변경사항**을 코드 예시와 함께
상세하게 정리한 것입니다. 변경의 동기, before/after 코드, 그리고 영향까지
포함합니다.

---

## 0. 변경 요약 (한눈에 보기)

| 영역 | 변경 횟수 | 핵심 효과 |
|------|----------|----------|
| `neurosim_utils.py` | 전면 재작성 | MLP_NeuroSim V3.0 알고리즘 정확 구현, 4가지 advanced mode 추가 |
| `models.py` | 전면 재작성 | Glorot-bounded 가중치 초기화로 CNN 정확도 ~10% → 95%+ 복구 |
| `data_loader.py` | path 수정 + ASL augmentation | 경로 hardcoding 제거, ASL test acc 향상 옵션 |
| `config.py` | 12개 옵션 추가 | LR/PSF/BN 기본값 튜닝 + V3 mode toggles |
| `train_eval.py` | `train_online()` 추가 + read tracking | V3.0 canonical online learning |
| `main.py` | V3 mode 인터랙티브 prompt | 터미널에서 4가지 모드 선택 가능 |
| `analysis_runners.py` | V3 mode 통합 | A/NL sweep도 V3 modes 지원, 에너지 column 추가 |
| `degradation_ui.py` | V3 mode controls + pair_mode dispatch | Streamlit UI에서 V3 modes 사용 가능 |
| `headless_runner.py` | 신규 | non-interactive CLI runner (CI/sweep용) |
| `verification_suite.py` | 신규 | (model × dataset) cross-product 회귀 테스트 |

---

## 1. `neurosim_utils.py` — 핵심 알고리즘 재작성

### 1.1 LTD curve fitting의 boundary condition

**문제**: 원본은 `B`를 자유 fit. 그 결과 boundary condition (P=Pmax에서 G_min/G_max)이 안 맞아 곡선이 rail 을 못 맞춤. V3.0은 `B`를 닫힌 형식으로 강제.

**Before (original):**
```python
def _fit_ltp(self):
    if self.use_p_start_offset:
        p0 = self.p_start_ltp
        def model(P, A, B):
            return B * (1 - np.exp(-(P - p0) / A)) + self.g_min_fit_scaled
    else:
        def model(P, A, B):
            return B * (1 - np.exp(-P / A)) + self.g_min_fit_scaled
    try:
        init = [np.mean(self.ltp_p_np), self.target_max - self.g_min_fit_scaled]
        params, _ = curve_fit(model, self.ltp_p_np, self.ltp_g_scaled, p0=init, maxfev=10000)
        return float(params[0]), float(params[1])  # (A, B) — both free
    except Exception as e:
        ...
```

**After (V3.0 canonical):**
```python
@staticmethod
def _B_from_A_scalar(A, Pmax):
    """Closed-form B such that G(Pmax) hits the opposite rail (V3.0 boundary)."""
    denom = 1.0 - float(np.exp(-Pmax / max(A, 1e-9)))
    if abs(denom) < 1e-12:
        return 0.0
    return 2.0 / denom   # target_range = 2 (-1 to +1)

def _fit_curve_for_A(self, P, G_scaled, start_scaled, ratio, ltp, pmax):
    if ltp:
        def model(P_, A):
            B = self._B_from_A_scalar(A, pmax)  # ← B 자동
            return B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled
    else:
        def model(P_, A):
            B = self._B_from_A_scalar(A, pmax)
            return -B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled
    # only A is fit; B is derived
    popt, _ = curve_fit(model, P_use, G_use, p0=[max(pmax/2, 1.0)], ...)
```

**영향**: G(P=0) = G_min, G(P=Pmax) = G_max가 정확히 보장됨. fit residual이 줄고 σ_c2c 추정이 더 정확해짐.

---

### 1.2 LTD PulseNum re-zeroing

**문제**: 원본 데이터에서 LTP는 PulseNum 1→50, LTD는 51→100. 원본 코드는 `p_start_force_zero` 옵션으로만 처리했고, 그 옵션을 끄면 LTD가 P=51부터 시작 → curve가 완전 어긋남.

**Before:**
```python
if self.use_p_start_offset and self.p_start_force_zero:
    self.p_start_ltp = 0.0
    self.p_start_ltd = 0.0
else:
    self.p_start_ltp = raw_p_start_ltp
    self.p_start_ltd = raw_p_start_ltd  # 51 → curve broken!
```

**After:**
```python
# ALWAYS rezero LTD's PulseNum
self.ltp_p_np = ltp_p_raw - ltp_p_raw.min()   # 1..50 → 0..49
self.ltd_p_np = ltd_p_raw - ltd_p_raw.min()   # 51..100 → 0..49
```

**영향**: 사용자 설정과 무관하게 항상 정확하게 fit. API 호환을 위해 옛 옵션은 no-op으로 유지.

---

### 1.3 σ_c2c 추정 방식

**문제**: 원본은 σ_c2c 추정 자체가 없었음. 추가했더니 fit residual을 쓰니까 first-pulse offset (실측 데이터의 "첫 펄스 점프")까지 포함돼서 σ=0.18 (현실의 3배) 과대 측정.

**Before (없었음):** σ_c2c 계산 기능 자체가 없음.

**Intermediate (residual 기반, 잘못된 추정):**
```python
def _estimate_sigma_c2c(self):
    """Residual std of the fitted curves on the scaled axis."""
    r1 = self.ltp_g_scaled - g_ltp_pred   # includes first-pulse jump
    r2 = self.ltd_g_scaled - g_ltd_pred
    sigma = float(np.std(np.concatenate([r1, r2])))  # σ = 0.18 (overestimate)
    return max(sigma, 0.0)
```

**After (diff 기반, 올바른 추정):**
```python
def _estimate_sigma_c2c(self):
    """Per-pulse cycle-to-cycle noise (std on the scaled axis).

    We measure step-to-step fluctuation, not the residual to the fitted
    curve — the residual is contaminated by the systematic first-pulse
    boundary mismatch and would massively overestimate true c2c.
    """
    def per_dir(P, G_scaled, ltp):
        ...
        r = G_scaled - G_pred
        dr = np.diff(r)  # diff removes the slow first-pulse offset
        return float(np.std(dr) / np.sqrt(2.0))  # diff of two indep Gaussians has var 2σ²
    s_ltp = per_dir(...)
    s_ltd = per_dir(...)
    return max(0.5 * (s_ltp + s_ltd), 0.0)  # σ = 0.06 (correct)
```

**영향**: σ_c2c가 0.18 → 0.06 (3x 작아짐). 그래서 노이즈 dominance가 사라지고 정확도가 회복.

---

### 1.4 Optimizer를 "stateless in P"로 재설계

**문제**: 원본 optimizer는 per-direction pulse counter (`pulse_states_ltp_n`, `pulse_states_ltd_n`)를 저장하고 매 step마다 G로부터 양쪽 다 재계산. LTP 한 번 쏘면 LTD counter까지 saturation으로 강제 이동하는 cross-talk bug 발생 → 발산.

**Before (broken state tracking):**
```python
class NeuroSimOptimizer(optim.Optimizer):
    def __init__(self, ...):
        # per-direction pulse states
        self.pulse_states_ltp_n = {}
        self.pulse_states_ltd_n = {}
        for p in params:
            self.pulse_states_ltp_n[id(p)] = fitter.p_of_g_ltp(p.data)
            self.pulse_states_ltd_n[id(p)] = fitter.p_of_g_ltd(p.data)

    def step(self, closure=None):
        for p in params:
            ...
            P_cur = self.pulse_states_ltp_n[pid][sel_mask]  # stale state!
            # ... update G_new ...
            p.data.copy_(G_new)
            # BUG: re-derive BOTH counters from G, even though only LTP fired
            g = p.data.clone()
            self.pulse_states_ltp_n[pid] = fitter.p_of_g_ltp(g)
            self.pulse_states_ltd_n[pid] = fitter.p_of_g_ltd(g)  # cross-talk!
```

**After (stateless in P, matches V3.0 `RealDevice::Write()`):**
```python
def _pulse_branch(self, sel_mask, pid, fitter, G_cur, G_tgt, G_new, psf, use_ltp, deg=False):
    A, B = self._get_AB(pid, use_ltp, fitter, deg)
    # invert G to find current P, then apply ΔP, then forward-map back to G
    P_cur = self._inverse_AB(G_cur[sel_mask], A, B, use_ltp, fitter)
    P_tgt = self._inverse_AB(G_tgt[sel_mask], A, B, use_ltp, fitter)
    dP = ((P_tgt - P_cur) * psf).nan_to_num(0.0)
    dP_q = self._stochastic_round(dP)
    P_next = (P_cur + dP_q).clamp(0.0, Pmax)
    G_branch = self._forward_AB(P_next, A, B, use_ltp, fitter)
    # σ_c2c noise on cells that ACTUALLY pulsed
    if self.use_c2c_noise and fitter.sigma_c2c > 0:
        actually_pulsed = (dP_q.abs() > 0.0)
        if actually_pulsed.any():
            noise = torch.randn_like(G_branch) * fitter.sigma_c2c
            G_branch = torch.where(actually_pulsed, G_branch + noise, G_branch)
    G_new[sel_mask] = G_branch.clamp(target_min, target_max)
    # energy accounting
    n = float(dP_q.abs().sum().item())
    if use_ltp: self.total_pulses_ltp += n
    else:       self.total_pulses_ltd += n
```

**영향**: cross-talk 사라짐. PSF가 1~300 모든 영역에서 학습 안정. CNN 정확도 ~10% → 95%+ 복구.

---

### 1.5 Differential pair mode (NEW)

**Before:** 없음 (단일 셀만)

**After:**
```python
def _step_pair_mixed(self, p, pid, grad, lr, psf):
    """V3.0-canonical differential pair using BOTH LTP and LTD."""
    Gp = self.G_pos[pid]
    Gn = self.G_neg[pid]
    delta_W = -lr * grad
    tmax = self.fitter.target_max

    # Per-element cell selection by headroom
    up_mask = grad < 0
    dn_mask = grad > 0
    prefer_ltp_pos = (tmax - Gp) >= Gn   # bigger headroom on G_pos LTP
    prefer_ltd_pos = Gp >= (tmax - Gn)   # bigger headroom on G_pos LTD

    ltp_pos_mask = up_mask & prefer_ltp_pos
    ltd_neg_mask = up_mask & (~prefer_ltp_pos)
    ltd_pos_mask = dn_mask & prefer_ltd_pos
    ltp_neg_mask = dn_mask & (~prefer_ltd_pos)

    # 4-way branching, normal + degraded cells separately
    ...

def _step_pair_ltp_only(self, p, pid, grad, lr, psf):
    """Burr/Boybat variant: LTP-only with periodic refresh."""
    ...
    # Refresh when both cells saturate
    sat = (Gp_new > 0.85*tmax) & (Gn_new > 0.85*tmax)
    if sat.any():
        delta = Gp_new[sat] - Gn_new[sat]
        Gp_new[sat] = delta.clamp(0, tmax)
        Gn_new[sat] = (-delta).clamp(0, tmax)
```

**영향**: 면적 2x 비용으로 W = G⁺ − G⁻ 표현 가능. asymmetric LTP/LTD 자연 cancel.

---

### 1.6 σ_d2d device-to-device variation (NEW)

**Before:** 모든 셀이 같은 A, B 사용.

**After:**
```python
def sample_per_cell_AB(self, shape, generator):
    """Sample per-cell A_LTP, A_LTD (log-normal) and matching B."""
    if self.sigma_d2d <= 0:
        return None
    A_LTP = self.A_LTP * torch.exp(self.sigma_d2d * torch.randn(shape, generator=generator))
    A_LTD = self.A_LTD * torch.exp(self.sigma_d2d * torch.randn(shape, generator=generator))
    A_LTP = A_LTP.clamp(min=1e-2)
    A_LTD = A_LTD.clamp(min=1e-2)
    B_LTP = self._B_from_A_tensor(A_LTP, self.Pmax_LTP)
    B_LTD = self._B_from_A_tensor(A_LTD, self.Pmax_LTD)
    return {"A_LTP": A_LTP, "B_LTP": B_LTP, "A_LTD": A_LTD, "B_LTD": B_LTD}

# In optimizer init:
if self.fitter.sigma_d2d > 0.0:
    self.per_cell_AB[pid] = self.fitter.sample_per_cell_AB(p.shape, generator=rng)

# In _pulse_branch: gathered per-cell A,B via the sel_mask
def _get_AB(self, pid, ltp, fitter, deg):
    store = self.per_cell_AB_deg if deg else self.per_cell_AB
    if pid in store:
        d = store[pid]
        return (d["A_LTP"], d["B_LTP"]) if ltp else (d["A_LTD"], d["B_LTD"])
    return (fitter.A_LTP, fitter.B_LTP) if ltp else (fitter.A_LTD, fitter.B_LTD)
```

**영향**: 같은 weight matrix 안에서도 셀마다 다른 A 사용 → 공간적 산포 (spatial variation) 모사 가능.

---

### 1.7 Energy accounting (NEW)

**Before:** 정확도만 추적.

**After:**
```python
DEFAULT_ENERGY = {
    "E_pulse_LTP_J": 1.0e-12,   # 1 pJ per write pulse
    "E_pulse_LTD_J": 1.0e-12,
    "E_read_J":      1.0e-15,   # 1 fJ per single-cell read MAC
    "cell_area_m2":  1.0e-12,   # 1 µm² per cell
}

class NeuroSimOptimizer:
    def __init__(self, ...):
        self.total_pulses_ltp = 0.0
        self.total_pulses_ltd = 0.0
        self.total_reads = 0
        self.total_weights = sum(p.numel() for p in params)

    def record_reads(self, n):
        self.total_reads += int(n)

    def energy_report(self):
        e_write = (self.total_pulses_ltp * E_pulse_LTP_J +
                   self.total_pulses_ltd * E_pulse_LTD_J)
        e_read = self.total_reads * E_read_J
        return {
            "total_pulses": self.total_pulses_ltp + self.total_pulses_ltd,
            "write_energy_J": e_write,
            "read_energy_J": e_read,
            "total_energy_J": e_write + e_read,
            "array_area_m2": self.total_weights * cell_area_m2 * (2 if pair else 1),
            ...
        }
```

**영향**: V3.0 C++ 본체와 동일한 energy/area 회계 가능. trade-off curve 작성 가능.

---

## 2. `models.py` — Weight 초기화 재작성

### 2.1 Glorot-bounded 초기화 (CNN 정확도 회복의 핵심)

**문제**: 원본은 모든 Conv/Linear를 `uniform_(-1, 1)` 로 초기화. 5×5×1 conv의 output std ≈ 2.9 → ReLU가 채널 절반을 죽임 → CNN 학습 불가능 (정확도 ~10%).

**Before:**
```python
def initialize_weights_uniform(model, weight_min, weight_max):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.uniform_(m.weight, a=weight_min, b=weight_max)   # ← range 전체
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**After:**
```python
def _bounded_glorot_(weight, w_min, w_max, gain=1.0):
    """Glorot-uniform init, clipped to [w_min, w_max]."""
    if weight.dim() < 2:
        bound = 1.0 / math.sqrt(weight.shape[0]) * gain
    else:
        fan_in = weight.shape[1] * (weight[0][0].numel() if weight.dim() > 2 else 1)
        fan_out = weight.shape[0] * (weight[0][0].numel() if weight.dim() > 2 else 1)
        bound = gain * math.sqrt(6.0 / max(fan_in + fan_out, 1))
    hw_half = 0.5 * (w_max - w_min)
    centre = 0.5 * (w_max + w_min)
    use_bound = min(bound, hw_half)   # never violate hardware range
    with torch.no_grad():
        weight.uniform_(centre - use_bound, centre + use_bound)

def initialize_weights_uniform(model, weight_min, weight_max, legacy_full_range=False):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if legacy_full_range:
                nn.init.uniform_(m.weight, weight_min, weight_max)   # opt-in
            else:
                _bounded_glorot_(m.weight, weight_min, weight_max)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

**예시 (SimpleNet fc1, 128×784):**
- Before: `weight.std()` ≈ 0.577 (uniform[-1,1])
- After: `weight.std()` ≈ 0.047 (Glorot bound ≈ 0.081)

**영향**: 
- Simple_CNN MNIST: 10% → **95.6%**
- Simple_CNN ASL: 10% → **80.8%**
- 모든 CNN 정상 학습 가능

---

### 2.2 Standard_CNN의 중복 dropout 제거

**Before:**
```python
self.fc = nn.Sequential(
    nn.Linear(64 * 7 * 7, 512),
    nn.BatchNorm1d(512), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256), nn.ReLU(),
    nn.Dropout(0.3),     # ← 두 번째 dropout (pulse noise와 중복)
    nn.Linear(256, num_classes)
)
```

**After:**
```python
self.fc = nn.Sequential(
    nn.Linear(64 * 7 * 7, 256),
    nn.BatchNorm1d(256), nn.ReLU(),
    nn.Dropout(0.25),    # single, mild dropout
    nn.Linear(256, num_classes),
)
```

**영향**: pulse-discretised weight에 dropout이 너무 많으면 학습 불안정. 한 개로 정리.

---

## 3. `data_loader.py` — 경로 및 augmentation

### 3.1 Hardcoded path 제거

**Before (`config.py`):**
```python
ASL_TRAIN_PATH = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/sign_mnist_train.csv"
ASL_TEST_PATH  = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/sign_mnist_test.csv"
```

**After:**
```python
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASL_TRAIN_PATH = os.path.join(_THIS_DIR, "sign_mnist_train.csv")
ASL_TEST_PATH  = os.path.join(_THIS_DIR, "sign_mnist_test.csv")
DEFAULT_DEVICE_XLSX = os.path.join(
    _THIS_DIR, "ZnO_Encap_Oven_48h_2 복사본", "0627_Pulse characteristic#1.xlsx"
)
```

---

### 3.2 ASL augmentation

**Before:**
```python
def get_asl_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ...
```

**After:**
```python
def get_asl_loaders(batch_size, augment: bool = True):
    """ASL train set has different humans / lighting than test set.
    Aug closes part of the gap."""
    normalise = transforms.Normalize((0.5,), (0.5,))
    if augment:
        train_t = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        train_t = transforms.Compose([transforms.ToTensor(), normalise])
    test_t = transforms.Compose([transforms.ToTensor(), normalise])
    ...
```

---

## 4. `config.py` — 기본값 튜닝 + V3 mode toggles

**Before:**
```python
# config.py
ASL_TRAIN_PATH = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/sign_mnist_train.csv"
ASL_TEST_PATH = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/sign_mnist_test.csv"

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001    # too low for the broken optimizer
PULSE_SCALING_FACTOR = 300  # too high; compensated for the bug
TARGET_RANGE = (-1.0, 1.0)

LTP_FIT_RATIO = 1.0
LTD_FIT_RATIO = 1.0
USE_BALANCED_TESTSET = False
ASL_USE_P_START_OFFSET = True
ASL_P_START_FORCE_ZERO = True
USE_BATCHNORM = False     # tanh saturates without BN
```

**After:**
```python
# config.py
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASL_TRAIN_PATH = os.path.join(_THIS_DIR, "sign_mnist_train.csv")
ASL_TEST_PATH  = os.path.join(_THIS_DIR, "sign_mnist_test.csv")
DEFAULT_DEVICE_XLSX = os.path.join(_THIS_DIR, "ZnO_Encap_Oven_48h_2 복사본",
                                   "0627_Pulse characteristic#1.xlsx")

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.01      # ← retuned for the fixed optimizer
PULSE_SCALING_FACTOR = 5  # ← retuned (no longer needs to be huge)
TARGET_RANGE = (-1.0, 1.0)

LTP_FIT_RATIO = 1.0
LTD_FIT_RATIO = 1.0
USE_BALANCED_TESTSET = False
ASL_USE_P_START_OFFSET = True
ASL_P_START_FORCE_ZERO = True
USE_BATCHNORM = True      # ← BN recommended, +2-3% accuracy

# V3.0 non-ideality controls
USE_C2C_NOISE = True
USE_CONDUCTANCE_DISCRETISATION = False
NUM_CONDUCTANCE_STATES = None

# V3.0 advanced modes (all OFF by default → previous behaviour unchanged)
USE_PAIR_MODE = False
PAIR_STRATEGY = "mixed"
SIGMA_D2D = 0.0
D2D_SEED = 0
USE_ONLINE_TRAINING = False
ONLINE_MICRO_BATCH = 1
ONLINE_MAX_SAMPLES_PER_EPOCH = None

# Energy / area accounting
ENERGY_PARAMS = {
    "E_pulse_LTP_J": 1.0e-12,
    "E_pulse_LTD_J": 1.0e-12,
    "E_read_J":      1.0e-15,
    "cell_area_m2":  1.0e-12,
}
```

---

## 5. `train_eval.py` — Online training + read tracking

### 5.1 Read event tracking (silent, always on)

**Before:** training loops just call `model(data)`, no energy accounting.

**After:**
```python
def _record_reads_if_possible(optimizer, n_samples):
    """Each forward reads every weight once per sample."""
    if hasattr(optimizer, "record_reads") and hasattr(optimizer, "total_weights"):
        optimizer.record_reads(int(n_samples) * int(optimizer.total_weights))

def train(model, device, train_loader, optimizer, epoch, loss_mode="ce"):
    ...
    for data, target in train_loader:
        ...
        optimizer.step()
        _record_reads_if_possible(optimizer, data.size(0))   # NEW
```

---

### 5.2 Online (batch=1) training mode (NEW)

```python
def _set_bn_eval(model):
    """Force every BN into eval mode (batch=1 in train mode is degenerate)."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval(); n += 1
    return n

def train_online(model, device, train_loader, optimizer, epoch, loss_mode="ce",
                 micro_batch=1, max_samples_per_epoch=None, progress_every=500):
    """V3.0-canonical per-sample training."""
    model.train()
    bn_count = _set_bn_eval(model)
    if bn_count:
        print(f"[online] forced {bn_count} BN layer(s) to eval mode")

    correct = total = sample_counter = 0
    for data_b, target_b in train_loader:
        data_b = data_b.to(device); target_b = target_b.to(device)
        for i in range(0, data_b.size(0), micro_batch):
            x = data_b[i:i + micro_batch]
            y = target_b[i:i + micro_batch]
            optimizer.zero_grad()
            output = model(x)
            loss = (nn.CrossEntropyLoss()(output, y) if loss_mode == "ce"
                    else F.nll_loss(output, y))
            loss.backward()
            optimizer.step()
            _record_reads_if_possible(optimizer, x.size(0))
            ...
            if max_samples_per_epoch and sample_counter >= max_samples_per_epoch:
                return ...
```

---

## 6. `main.py` — Interactive V3 mode prompt

**Before:** dataset → model → analysis 3 단계 prompt만.

**After:** 그 뒤에 V3 mode 4-step prompt 추가:
```python
def select_v3_modes():
    """터미널에서 V3.0 advanced mode 들을 켤지/끌지 묻기."""
    print("\n" + "=" * 50)
    print("V3.0 advanced modes (기본값 = 전부 OFF, 이전 결과 그대로)")
    print("=" * 50)
    pair = _prompt_yes_no("1) Differential pair (G⁺ − G⁻) mode 사용?", default=False)
    pair_strategy = "mixed"
    if pair:
        print("   ↳ pair-update 전략:")
        print("     [1] mixed     (V3.0 canonical: LTP+LTD, headroom 기반 셀 선택) (recommended)")
        print("     [2] ltp_only  (Burr/Boybat 변형: LTP만 + 주기적 refresh)")
        sel = input("     선택 [1/2, default 1]: ").strip()
        pair_strategy = "ltp_only" if sel == "2" else "mixed"
    use_d2d = _prompt_yes_no("2) Device-to-device variation (σ_d2d) 사용?", default=False)
    sigma_d2d = (_prompt_float("   ↳ σ_d2d 값", default=0.10)
                 if use_d2d else 0.0)
    online = _prompt_yes_no("3) Online (batch=1) training mode 사용?", default=False)
    energy = _prompt_yes_no("4) Pulse-energy 회계 결과 출력?", default=True)
    return {
        "pair_mode": pair,
        "pair_strategy": pair_strategy,
        "sigma_d2d": float(sigma_d2d),
        "online": online,
        "energy": energy,
    }
```

그 다음 학습 시 모든 v3_modes를 optimizer로 전달:
```python
optimizer = NeuroSimOptimizer(
    model.parameters(), config.LEARNING_RATE, fitter, config.PULSE_SCALING_FACTOR,
    use_c2c_noise=getattr(config, "USE_C2C_NOISE", True),
    use_discretisation=getattr(config, "USE_CONDUCTANCE_DISCRETISATION", False),
    pair_mode=bool(v3_modes.get("pair_mode", False)),
    pair_strategy=str(v3_modes.get("pair_strategy", "mixed")),
    energy_params=getattr(config, "ENERGY_PARAMS", None),
    d2d_seed=getattr(config, "D2D_SEED", 0),
)
```

그리고 학습 끝에 energy report 출력:
```python
if v3_modes.get("energy", True):
    rep = optimizer.energy_report()
    print(f"  LTP pulses        : {rep['total_pulses_ltp']:.0f}")
    print(f"  LTD pulses        : {rep['total_pulses_ltd']:.0f}")
    print(f"  Write energy      : {rep['write_energy_J']*1e6:.4f}  µJ")
    print(f"  Read energy       : {rep['read_energy_J']*1e6:.4f}  µJ")
    print(f"  Crossbar area     : {rep['array_area_m2']*1e12:.2f}  µm²")
```

---

## 7. `analysis_runners.py` — V3 modes propagation

**Before:** A-vs-acc / NL-vs-acc 함수가 V3 modes 무시.

**After:**
```python
def run_analysis(file_paths, mode, train_loader, test_loader,
                 model_factory, loss_mode, dataset_label="",
                 v3_modes: dict | None = None):   # NEW arg
    if v3_modes is None:
        v3_modes = {}
    pair_mode = bool(v3_modes.get("pair_mode", False))
    pair_strategy = str(v3_modes.get("pair_strategy", "mixed"))
    sigma_d2d = float(v3_modes.get("sigma_d2d", 0.0))

    for f_path in file_paths:
        fitter = NeuroSimFitter(..., sigma_d2d=sigma_d2d)   # ← d2d 적용
        optimizer = NeuroSimOptimizer(
            ..., pair_mode=pair_mode, pair_strategy=pair_strategy,
            energy_params=getattr(config, "ENERGY_PARAMS", None),
            d2d_seed=getattr(config, "D2D_SEED", 0),
        )
        ...
        final_acc = float(epoch_accs[-1])
        energy = optimizer.energy_report()    # ← 에너지도 추적
        results.append((os.path.basename(f_path), metric_val, final_acc,
                        energy["total_pulses"], energy["total_energy_J"] * 1e6))

    # 결과 엑셀에 에너지 column 추가
    df = pd.DataFrame(results, columns=["File", mode, "Accuracy", "TotalPulses", "Energy_uJ"])
```

---

## 8. `degradation_ui.py` — Streamlit UI 확장

### 8.1 Hardcoded path 폴백

**Before:**
```python
DEFAULT_NORMAL_XLSX = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/정상소자.xlsx"
DEFAULT_DEGRADED_XLSX = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/열화소자.xlsx"
```

**After:** 옛 경로가 있으면 그것, 없으면 V2 폴더의 기본 xlsx:
```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_V2_XLSX = getattr(config, "DEFAULT_DEVICE_XLSX",
                           os.path.join(_THIS_DIR, "ZnO_Encap_Oven_48h_2 복사본",
                                        "0627_Pulse characteristic#1.xlsx"))
_LEGACY_N = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/정상소자.xlsx"
_LEGACY_D = "/Users/parkhyungbin/Desktop/연구실/ANN/neurosim code (함수분리)/열화소자.xlsx"

DEFAULT_NORMAL_XLSX = _LEGACY_N if os.path.exists(_LEGACY_N) else _DEFAULT_V2_XLSX
DEFAULT_DEGRADED_XLSX = _LEGACY_D if os.path.exists(_LEGACY_D) else _DEFAULT_V2_XLSX
```

---

### 8.2 V3 mode sidebar (NEW)

**Before:** Fitter option만:
```python
st.subheader("Fitter 옵션")
use_offset = st.checkbox("use_p_start_offset", value=True)
force_zero = st.checkbox("p_start_force_zero", value=True)
```

**After:** V3 mode들 전부 노출:
```python
st.subheader("Fitter 옵션")
use_offset = st.checkbox("use_p_start_offset (API compat, no-op)", value=True)
force_zero = st.checkbox("p_start_force_zero (API compat, no-op)", value=True)

st.subheader("V3.0 advanced modes")
v3_pair = st.checkbox("Differential pair (G⁺ − G⁻)", value=False,
                      help="셀당 2개의 conductance 셀로 weight 표현. 면적 2배.")
if v3_pair:
    v3_pair_strategy = st.radio(
        "Pair update 전략",
        ["mixed (V3.0 canonical, LTP+LTD)", "ltp_only (Burr/Boybat 변형)"],
    ).split()[0]
v3_d2d_n = st.number_input("σ_d2d (정상 fitter)", value=0.0, ...)
v3_d2d_d = st.number_input("σ_d2d (열화 fitter)", value=0.0, ...)
v3_online = st.checkbox("Online (batch=1) training mode", value=False)
if v3_online:
    v3_online_mb = st.number_input("online micro_batch", min_value=1, value=1)
    v3_online_cap = st.number_input("online max_samples_per_epoch (0=무제한)",
                                     min_value=0, value=0)
```

---

### 8.3 `pair_mode` + degradation mask 디스패치 (CRITICAL FIX)

**Before:** `_step_pair_*` 함수가 `self.masks`와 `self.degraded_fitter`를 무시 → pair_mode를 켜면 degradation_ui의 열화 모델이 silent하게 동작하지 않음.

**After:** 새 helper `_split_normal_degraded` 추가, pair step 두 가지 모두 normal/degraded 분기:
```python
def _split_normal_degraded(self, pid, G_cur):
    """Return (normal_mask, degraded_mask) tensors matching G_cur shape."""
    cell_mask = self.masks.get(pid)
    if cell_mask is None or self.degraded_fitter is None:
        normal = torch.ones_like(G_cur, dtype=torch.bool)
        degraded = torch.zeros_like(G_cur, dtype=torch.bool)
    else:
        cm = cell_mask.to(device=G_cur.device).bool()
        normal = cm
        degraded = ~cm
    return normal, degraded

# In _step_pair_mixed:
normal, degraded = self._split_normal_degraded(pid, Gp)
# NORMAL cells → nominal fitter
self._pulse_branch(ltp_pos_mask & normal, pid, f_n, ..., use_ltp=True)
self._pulse_branch(ltd_pos_mask & normal, pid, f_n, ..., use_ltp=False)
... (모두 4-way)
# DEGRADED cells → degraded fitter
if f_d is not None:
    self._pulse_branch(ltp_pos_mask & degraded, pid, f_d, ..., use_ltp=True, deg=True)
    self._pulse_branch(ltd_pos_mask & degraded, pid, f_d, ..., use_ltp=False, deg=True)
    ... (모두 4-way)
```

**영향**: 이제 degradation_ui에서 pair_mode 켜도 degraded_fitter가 정상 동작.

---

### 8.4 Excel export에 V3 metadata 추가

**Before:**
```python
row_data = {
    "Model": ...,
    "Target Layers": ...,
    "Mode": ...,
    "Degraded Ratio (%)": ...,
    "Final Acc (%)": ...
}
```

**After (V3 column 추가):**
```python
row_data = {
    "Model": ..., "Target Layers": ..., ...
    "Final Acc (%)": ...,
    "Pair Mode": r.get("pair_mode", False),
    "Pair Strategy": r.get("pair_strategy", "-"),
    "Online Training": r.get("use_online", False),
    "LTP Pulses": int(r.get("pulses_ltp", 0)),
    "LTD Pulses": int(r.get("pulses_ltd", 0)),
    "Energy (µJ)": r.get("energy_uJ", 0),
    "Area (µm²)": r.get("area_um2", 0),
}
```

---

## 9. 신규 파일

### 9.1 `headless_runner.py` (신규)
```python
# CLI runner for sweeps / CI.
# All V3 modes exposed as flags.
python3 headless_runner.py --model Simple_CNN --dataset MNIST --epochs 2 \
        --pair-mode --pair-strategy mixed --sigma-d2d 0.15 \
        --online --online-micro-batch 8 --online-max-samples 30000 \
        --energy-report
```

### 9.2 `verification_suite.py` (신규)
- 7 (model) × 2 (dataset) cross-product 자동 실행
- JSON + CSV로 저장
- 매 epoch acc, energy, σ 정보 다 기록

### 9.3 Reports
- `RESEARCH_NOTES.md` — V3.0 알고리즘 분석 + audit
- `REFACTOR_REPORT.md` — 1차 리팩토링 (optimizer + init 수정)
- `V3_MODES_REPORT.md` — 4가지 advanced modes
- `PAIR_STRATEGY_REPORT.md` — pair mode mixed vs ltp_only
- `CHANGES.md` — 이 문서

---

## 10. 정확도 회귀 테스트 (regression)

모든 V3 mode OFF + 기존 hyperparameter → 결과가 **bit-for-bit** 동일해야 함.

| Test                                | Expected           | Actual             | OK? |
|-------------------------------------|--------------------|--------------------|-----|
| SimpleNet MNIST (seed 0, 3 epochs) | [88.93, 89.5, 90.24] | [88.93, 89.5, 90.24] | ✅ |
| Simple_CNN MNIST (seed 0, 2 epochs) | [95.02, 95.62]    | [95.02, 95.62]      | ✅ |

V3 modes 켜도 ±1% 이내 유지:

| Configuration                                 | Best test acc | vs baseline |
|-----------------------------------------------|---------------|-------------|
| SimpleNet MNIST baseline                      | 90.24 %       | —           |
| SimpleNet MNIST + pair=mixed                  | 90.56 %       | +0.32 %     |
| SimpleNet MNIST + pair=ltp_only               | 90.29 %       | +0.05 %     |
| SimpleNet MNIST + σ_d2d=0.15                  | 89.42 %       | −0.82 %     |
| SimpleNet MNIST + pair=mixed + σ_d2d=0.15     | 89.73 %       | −0.51 %     |
| Simple_CNN MNIST baseline                     | 95.62 %       | —           |
| Simple_CNN MNIST + pair=mixed                 | 96.15 %       | +0.53 %     |
| Simple_CNN ASL baseline                       | 79.53 %       | —           |
| Simple_CNN ASL + pair=mixed                   | 80.10 %       | +0.57 %     |
| Simple_CNN ASL + σ_d2d=0.15                   | 80.05 %       | +0.52 %     |
| degradation_ui (single, 30 % deg same xlsx)   | 89.55 %       | (baseline)  |
| degradation_ui (pair=mixed, 30 % deg same)    | 89.71 %       | +0.16 %     |

**모든 mode에서 ≥ 88% MLP / ≥ 95% CNN MNIST 유지됨.** ✅

---

## 11. 파일별 변경 라인 수 요약

| File                       | Lines before | Lines after | Δ      | 주된 변경                      |
|----------------------------|--------------|-------------|--------|--------------------------------|
| `neurosim_utils.py`        | ~360         | ~550        | +190   | V3 algorithm + 4 modes         |
| `models.py`                | ~205         | ~210        | +5     | _bounded_glorot_ 추가          |
| `data_loader.py`           | ~190         | ~200        | +10    | ASL augment, path 자동화       |
| `config.py`                | ~25          | ~50         | +25    | V3 toggles                     |
| `train_eval.py`            | ~145         | ~210        | +65    | train_online + read tracking   |
| `main.py`                  | ~325         | ~390        | +65    | V3 prompt + energy report      |
| `analysis_runners.py`      | ~120         | ~130        | +10    | v3_modes 통합                  |
| `degradation_ui.py`        | ~635         | ~700        | +65    | V3 sidebar + pair dispatch fix |
| `headless_runner.py`       | 0 (신규)     | ~200        | +200   | 신규                           |
| `verification_suite.py`    | 0 (신규)     | ~110        | +110   | 신규                           |

---

## 12. 사용법 요약

### 기존 워크플로 (변경 없음)
```bash
python3 main.py
# dataset, model, analysis 순서대로 선택
# 추가로 V3 modes 4가지 yes/no
```

### CLI 자동화
```bash
# Baseline
python3 headless_runner.py --model SimpleNet --dataset MNIST --epochs 10

# pair mode + d2d + 에너지 보고
python3 headless_runner.py --model Simple_CNN --dataset MNIST --epochs 3 \
        --pair-mode --pair-strategy mixed --sigma-d2d 0.15 --energy-report

# Online (V3.0 canonical 학습)
python3 headless_runner.py --model SimpleNet --dataset MNIST --epochs 2 \
        --online --online-micro-batch 8 --online-max-samples 30000 --lr 2e-3
```

### Streamlit UI
```bash
streamlit run degradation_ui.py
# sidebar에 V3 modes 컨트롤 추가됨
```

### Python API
```python
from neurosim_utils import NeuroSimFitter, NeuroSimOptimizer

fitter = NeuroSimFitter(ltp_df, ltd_df, target_range=(-1,1),
                        sigma_d2d=0.15)   # device-to-device variation
opt = NeuroSimOptimizer(
    model.parameters(), lr=1e-2, fitter=fitter, pulse_scaling_factor=5,
    use_c2c_noise=True,
    pair_mode=True, pair_strategy="mixed",
    energy_params={"E_pulse_LTP_J": 1.5e-12, ...},
)
# ... train ...
print(opt.energy_report())
```
