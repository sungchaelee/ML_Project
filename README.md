# 202221016 이성채 ML_Project
건국대학교 컴퓨터공학과 기계학습 ML 기말 프로젝트 - CardioCare  종단간 머신러닝 시스템 구축

# CardioCare — 종단간 심장병 예측 ML 시스템

임상 데이터로 심장병 발병 가능성을 예측해 심장 전문의의 의사결정을 **보조**하는
end-to-end ML 시스템이다. 이 시스템은 의사를 돕는 도구이며 **단독으로 진단하지
않는다 (inform, not decide).**

## 데이터셋
UCI Heart Disease (https://archive.ics.uci.edu/dataset/45/heart+disease) 통합본.
4개 출처(Cleveland·Hungary·Switzerland·VA)의 `processed.*.data`를 합쳐 920행,
타깃은 0(정상)/1(심장병)으로 이진화. 원본 파일은 `data/`에 포함되어 별도 다운로드 불필요.

## 저장소 구조

├── data/                       # UCI 원본(processed.*.data) + sample_input.csv
├── notebooks/01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py        # 로드·정리·전처리 파이프라인
│   ├── train.py                # 학습·MLflow·튜닝·최종 모델 저장
│   ├── inference.py            # 추론 + 로깅
│   └── monitor.py              # 드리프트 탐지·모니터링
├── tests/test_pipeline.py
├── models/final_model.pkl      # 저장된 최종 모델
├── mlruns/                     # MLflow 실험 기록
├── logs/                       # 추론 로그 + 드리프트 그래프
├── Dockerfile
├── requirements.txt            # 버전 고정
├── requirements-inference.txt  # Docker용 경량 의존성
├── .github/workflows/ci.yml
├── report.pdf
└── README.md

## 환경 설정
Python 3.12 권장 (3.10+).
```bash
git clone <저장소 URL>
cd <저장소>

# 가상환경
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

## 재현 순서

**1. 모델 학습 (MLflow 기록 + 최종 모델 저장)**
```bash
python src/train.py
```
→ `models/final_model.pkl` 생성, `mlruns/`에 4개 실험(모델 3종 + 튜닝 최종) 기록.

**2. MLflow 실험 보기**
```bash
# Windows
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri ./mlruns
# Mac/Linux
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
```
→ http://127.0.0.1:5000 에서 파라미터·지표·모델 아티팩트 확인.

**3. 단위 테스트**
```bash
python -m unittest discover -s tests -v
```

**4. Docker (추론 패키징)**
```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```
→ `data/sample_input.csv`에 대한 예측 출력.

**5. 모니터링 / 드리프트**
```bash
python src/monitor.py
```
→ KS 검정으로 드리프트 특성 flag + balanced accuracy 저하 보고 + `logs/drift_monitoring.png` 생성.

## 최종 모델
SVC (RBF). 임상적으로 false negative 최소화를 우선해 recall 기준으로 선택·튜닝.
상세 정당화는 `report.pdf` 참조.

## 재현성
- 랜덤 시드 고정 (`random_state=42`).
- `requirements.txt` 버전 고정.
- 데이터 결정론적 로드 (`src/preprocessing.py: load_raw_data`).
- 모든 데이터 학습 변환(대치·스케일링·특성선택)은 학습 fold에만 fit → 데이터 누수 방지.

## AI 도구 사용
AI 도구 사용 내역은 `report.pdf` 부록에 공개함.