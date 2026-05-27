# config.py
import os

# --- 경로 설정 (ASL 데이터셋 경로 - V2 폴더 안의 CSV 사용) ---
# auto-discover sign_mnist CSVs relative to this file so the code works on
# any machine without hard-coded absolute paths.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASL_TRAIN_PATH = os.path.join(_THIS_DIR, "datasets", "sign_mnist_train.csv")
ASL_TEST_PATH  = os.path.join(_THIS_DIR, "datasets", "sign_mnist_test.csv")

# Default device characteristic for non-interactive runs.
DEFAULT_DEVICE_XLSX = os.path.join(
    _THIS_DIR, "datasets", "zno_encap_48h", "0627_Pulse characteristic#1.xlsx"
)

# --- 하이퍼파라미터 ---
# Defaults retuned for the refactored stateless-P NeuroSim optimizer.
# Old defaults (lr=1e-4, psf=300) corresponded to a saturated-pulse regime
# that worked only by accident with the previous buggy P-counter code.
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.01
PULSE_SCALING_FACTOR = 5
TARGET_RANGE = (-1.0, 1.0)

# --- NeuroSim 피팅 옵션 ---
LTP_FIT_RATIO = 1.0  # 데이터의 100% 사용
LTD_FIT_RATIO = 1.0

# --- 테스트셋 옵션 (True: 클래스별 최소 개수(892)로 맞춤, False: 전체 사용) ---
USE_BALANCED_TESTSET = False
ASL_USE_P_START_OFFSET = True
ASL_P_START_FORCE_ZERO = True

# --- SimpleNet(MNIST MLP) 구조 옵션 ---
# True: fc1 뒤에 BatchNorm1d 사용 (권장; ≥90% on ZnO_Encap_48h)
# False: BN 제거 (legacy; ~88%)
USE_BATCHNORM = True

# --- NeuroSim non-ideality controls (V3.0 style) ---
USE_C2C_NOISE = True
USE_CONDUCTANCE_DISCRETISATION = False
NUM_CONDUCTANCE_STATES = None   # e.g. 64 for 6-bit weight, None = continuous

# --- V3.0 advanced modes (all OFF by default - default behaviour unchanged) ---
USE_PAIR_MODE = False           # differential pair (G⁺ − G⁻) per weight
PAIR_STRATEGY = "mixed"         # "mixed" (V3.0 canonical, uses LTP+LTD) or
                                # "ltp_only" (Burr/Boybat variant, LTP only + refresh)
SIGMA_D2D = 0.0                 # device-to-device variation (relative, log-normal on A)
D2D_SEED = 0                    # RNG seed for D2D sampling

# Online (batch=1) training mode
USE_ONLINE_TRAINING = False
ONLINE_MICRO_BATCH = 1          # 1 = pure online; 4-8 = "almost online" for speed
ONLINE_MAX_SAMPLES_PER_EPOCH = None   # cap per-epoch samples; None = no cap

# Energy / area accounting (per-cell)
ENERGY_PARAMS = {
    "E_pulse_LTP_J": 1.0e-12,   # 1 pJ per LTP write pulse  (placeholder; tune from datasheet)
    "E_pulse_LTD_J": 1.0e-12,
    "E_read_J":      1.0e-15,   # 1 fJ per single-cell read MAC
    "cell_area_m2":  1.0e-12,   # 1 µm² per cell
}