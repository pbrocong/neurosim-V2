# NeuroSim V2

ZnO 멤리스터 기반 in-memory-computing 학습 시뮬레이터.
펄스 도메인에서 신경망을 학습시키고, 소자 비이상성(C2C/D2D 노이즈, 양자화),
차분 페어, 온라인 학습, 에너지/면적을 시뮬레이션합니다.

---

## 📁 폴더 구조 — 뭐가 어디 있나

```
neurosim code V2/
│
├── simulator/        ⚙️  핵심 엔진 (시뮬레이터 라이브러리)
│   ├── config.py             하이퍼파라미터 기본값 + 경로 설정
│   ├── models.py             신경망 (SimpleNet, CNN, LeNet5, VGG, AlexNet, ResNet18)
│   ├── neurosim_utils.py     NeuroSimFitter / NeuroSimOptimizer (펄스 도메인 핵심)
│   ├── data_loader.py        MNIST/Fashion/KMNIST/CIFAR/ASL 로더
│   ├── train_eval.py         train / train_online / test
│   ├── analysis_runners.py   A-값 / NL 분석 루틴
│   ├── headless_runner.py    배치 CLI (인자로 1회 학습)
│   ├── verification_suite.py 모델×데이터셋 스윕 검증
│   ├── main.py               대화형 CLI (레거시)
│   └── degradation_ui.py     Streamlit 열화 UI (레거시)
│
├── webapp/           🌐  Hugging Face / Gradio 웹앱  (← 모든 웹 코드는 여기)
│   ├── app.py                Gradio 진입점 (5탭)
│   ├── gradio_app/           스키마·러너·위젯·설정IO
│   ├── requirements.txt      웹앱 의존성
│   ├── README.md             HF Spaces 설정(frontmatter) + 앱 설명
│   └── DEPLOY.md             HF Spaces 배포 방법
│
├── tests/            ✅  검증
│   └── parity_check.py       웹앱 결과 == 엔진 직접 실행 결과 (11케이스)
│
├── datasets/         📦  입력 데이터 (sign_mnist CSV, ZnO 소자 xlsx)
├── data/             📦  torchvision 자동 다운로드 캐시 (git 제외)
├── docs/             📄  리포트 / 연구 노트 / docx
└── results/          📊  실행 결과물 (xlsx/csv/json)
```

핵심만 기억하면:
- **`simulator/`** = 계산 엔진 (라이브러리)
- **`webapp/`** = 웹 인터페이스 (Hugging Face에 올라가는 부분)
- **나머지** = 데이터·문서·결과·테스트

---

## 🚀 실행 방법

모든 명령은 **이 폴더(레포 루트)에서** 실행하세요.

### 웹앱 (권장)
```bash
pip install -r webapp/requirements.txt
python webapp/app.py
# → http://127.0.0.1:7860
```
모든 하이퍼파라미터를 GUI로 조정하고, 학습·스윕·열화·에너지 분석을 탭으로 실행합니다.

### 엔진 직접 실행 (CLI)
```bash
# 1회 학습 + 에너지 리포트
python simulator/headless_runner.py --model SimpleNet --dataset MNIST --epochs 5 --energy-report

# 모델×데이터셋 스윕 검증
python simulator/verification_suite.py --epochs 3
```

### 결과 일치 검증
```bash
python tests/parity_check.py
# 웹앱 경로와 엔진 직접 실행이 bit-for-bit 같은지 11개 조합으로 확인
```

---

## 🔬 검증 상태

`tests/parity_check.py`가 웹앱(`gradio_app.runners`)과 원본 엔진(`headless_runner`)을
같은 시드·하이퍼파라미터로 돌려 **정확도 + 에너지 + 피팅 파라미터**가 완전히
일치하는지 확인합니다. 기본 모델 5종 + 고급 모드(차분 페어 mixed/ltp_only, σ_d2d,
양자화, 온라인, C2C off) 총 11케이스 모두 통과.

---

## ⚠️ 참고

- 데이터셋 CSV(`sign_mnist_*.csv`, ~104 MB)와 `data/` 캐시는 git에 포함되지 않습니다.
  ASL을 쓰려면 `datasets/`에 CSV를 두세요.
- `config.py`의 에너지 파라미터(`E_pulse`, `E_read`, `cell_area`)는 placeholder입니다.
  실제 소자 datasheet 값으로 바꿔야 절대값이 의미를 가집니다 (지금은 상대비교용).
- Hugging Face 배포는 [webapp/DEPLOY.md](webapp/DEPLOY.md) 참고.
