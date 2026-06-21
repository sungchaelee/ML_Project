"""
CardioCare 전처리 모듈 (§5.1)

- clean_data():       데이터에서 '학습'하지 않는 결정론적 정리 (누수 안전)
- build_preprocessor(): 결측 대치 sklearn 파이프라인 (train에만 fit → 누수 방지)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 0이 생리학적으로 불가능한 컬럼 (숨은 결측)
ZERO_AS_MISSING = ["chol", "trestbps"]

# 과반 결측이라 드롭하는 컬럼
DROP_COLS = ["ca", "thal"]

# 대치 전략별 특성 그룹 (ca, thal 드롭 후 기준)
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
TARGET = "target"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """데이터에서 학습하지 않는 결정론적 정리. 누수 위험 없음.
    train/test 분할 '이전'에 적용해도 안전한 단계만 포함한다.
    """
    df = df.copy()

    # 1) 숨은 결측: 불가능한 0을 NaN으로
    for col in ZERO_AS_MISSING:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # 2) 중복 행 제거 (같은 행이 train/test 양쪽에 들어가는 누수도 예방)
    df = df.drop_duplicates().reset_index(drop=True)

    # 3) 고결측 컬럼 드롭
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 4) 빈(전부 NaN)/상수(값 1개) 컬럼 제거 (타깃은 보호)
    for col in list(df.columns):
        if col == TARGET:
            continue
        if df[col].isna().all() or df[col].nunique(dropna=True) <= 1:
            df = df.drop(columns=col)

    return df


def build_preprocessor() -> ColumnTransformer:
    """결측 대치 파이프라인 (아직 fit 안 함).
    - 연속형: 중앙값(median)  → 이상치에 robust
    - 범주/이진형: 최빈값(most_frequent)
    실제 fit은 §5.2에서 '학습 데이터에만' 수행 → 데이터 누수 방지.
    """
    continuous_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_pipe, CONTINUOUS_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # 명시한 컬럼만 사용
    )
    return preprocessor