"""
CardioCare 추론 모듈
실행: python src/inference.py <입력CSV경로>
"""
import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import balanced_accuracy_score

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"
MODEL_VERSION = "cardiocare-svc-1.0"

FEATURE_RANGES = {
    "age": (0, 120), "trestbps": (0, 300), "chol": (0, 600),
    "thalach": (0, 250), "oldpeak": (-5, 10),
}


def get_logger():
    """logs/inference.log 파일에 기록하는 로거."""
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("cardiocare_inference")
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "inference.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def prepare(df):
    df = df.copy()
    for col in ["chol", "trestbps"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def validate_input_ranges(df):
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


def predict_with_logging(df, model=None, y_true=None):
    """예측 + 추론 로그 기록 (타임스탬프·모델버전·입력shape·예측값·정답)."""
    logger = get_logger()
    model = model or load_model()
    preds = model.predict(prepare(df))
    msg = (f"model_version={MODEL_VERSION} | input_shape={df.shape} | "
           f"predictions={preds.tolist()}")
    if y_true is not None:
        bal_acc = balanced_accuracy_score(y_true, preds)
        msg += f" | actual_provided=True | balanced_accuracy={bal_acc:.3f}"
    logger.info(msg)
    return preds


def main():
    if len(sys.argv) < 2:
        print("사용법: python src/inference.py <입력CSV경로>")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    model = load_model()
    preds = predict_with_logging(df, model)  
    probs = model.predict_proba(prepare(df))[:, 1]
    for i, (p, pr) in enumerate(zip(preds, probs)):
        label = "심장병 의심" if p == 1 else "정상"
        print(f"row {i}: prediction={int(p)} ({label}), prob_disease={pr:.3f}")


if __name__ == "__main__":
    main()