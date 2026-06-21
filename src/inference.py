"""
CardioCare 추론 모듈 (§5.3)
실행: python src/inference.py <입력CSV경로>
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"

# 임상적으로 타당한 입력 범위
FEATURE_RANGES = {
    "age": (0, 120), "trestbps": (0, 300), "chol": (0, 600),
    "thalach": (0, 250), "oldpeak": (-5, 10),
}


def load_model(path=MODEL_PATH):
    """저장된 최종 모델을 불러온다."""
    return joblib.load(path)


def prepare(df):
    """추론용 최소 전처리: 불가능한 0을 NaN으로 (학습 때와 동일 규칙)."""
    df = df.copy()
    for col in ["chol", "trestbps"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def validate_input_ranges(df):
    """범위를 벗어난 특성을 {컬럼: 개수}로 반환. 비어 있으면 모두 정상."""
    issues = {}
    for col, (lo, hi) in FEATURE_RANGES.items():
        if col in df.columns:
            n_bad = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_bad > 0:
                issues[col] = n_bad
    return issues


def predict(df, model=None):
    model = model or load_model()
    return model.predict(prepare(df))


def predict_proba(df, model=None):
    model = model or load_model()
    return model.predict_proba(prepare(df))


def main():
    if len(sys.argv) < 2:
        print("사용법: python src/inference.py <입력CSV경로>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    model = load_model()
    preds = model.predict(prepare(df))
    probs = model.predict_proba(prepare(df))[:, 1]
    for i, (p, pr) in enumerate(zip(preds, probs)):
        label = "심장병 의심" if p == 1 else "정상"
        print(f"row {i}: prediction={int(p)} ({label}), prob_disease={pr:.3f}")


if __name__ == "__main__":
    main()