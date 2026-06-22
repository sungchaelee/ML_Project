"""
CardioCare 모니터링 & 드리프트 탐지
실행: python src/monitor.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from preprocessing import load_raw_data, clean_data, TARGET
from inference import load_model, predict, predict_with_logging

RANDOM_STATE = 42
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def make_drifted(X):
    d = X.copy()
    m = d["chol"].mean()
    d["chol"] = m + (d["chol"] - m) * 1.5 + 50
    d["thalach"] = d["thalach"] - 25
    d["oldpeak"] = d["oldpeak"] + 1.5
    return d


def ks_report(reference, candidate, alpha=0.05):
    """각 연속형 특성에 대해 기준 분포 vs 후보 분포 KS 검정."""
    rows = []
    for f in CONTINUOUS:
        stat, p = ks_2samp(reference[f].dropna(), candidate[f].dropna())
        rows.append({"feature": f, "ks_stat": round(stat, 3),
                     "p_value": round(p, 4), "drift(p<0.05)": p < alpha})
    return pd.DataFrame(rows)

def drift_timeseries(model, X_test, y_test, save_path):
    """드리프트를 점점 키우며 balanced accuracy 변화를 시계열로 그린다."""
    factors = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    base = datetime(2026, 1, 1)
    timestamps, scores = [], []
    for i, factor in enumerate(factors):
        d = X_test.copy()
        m = d["chol"].mean()
        d["chol"] = m + (d["chol"] - m) * (1 + 0.5 * factor) + 50 * factor
        d["thalach"] = d["thalach"] - 25 * factor
        d["oldpeak"] = d["oldpeak"] + 1.5 * factor
        scores.append(balanced_accuracy_score(y_test, predict(d, model)))
        timestamps.append(base + timedelta(weeks=i)) 

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, scores, marker="o", color="steelblue")
    plt.axhline(0.70, color="red", linestyle="--", label="경고 임계값 0.70")
    plt.title("Balanced Accuracy over time (drift increasing)")
    plt.xlabel("Time (synthetic)")
    plt.ylabel("Balanced Accuracy")
    plt.ylim(0.4, 0.85)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"\n시계열 그래프 저장 완료: {save_path}")


def main():
    # 1) 로드 + 정리 + 분할 (train.py와 동일 시드)
    df = clean_data(load_raw_data())
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = load_model()

    # 모델이 실제 사용하는 특성 확인
    feat = model.named_steps["preprocess"].get_feature_names_out()
    mask = model.named_steps["select"].get_support()
    selected = [f.split("__")[-1] for f, m in zip(feat, mask) if m]
    print("모델이 사용하는 특성:", selected, "\n")

    # 2) 드리프트된 테스트셋 생성
    X_test_drift = make_drifted(X_test)

    # 3) KS 검정: 학습 vs 원본 / 학습 vs 드리프트
    print("=== KS 검정: 학습 vs 원본 테스트 (드리프트 없음 기대) ===")
    print(ks_report(X_train, X_test).to_string(index=False))
    print("\n=== KS 검정: 학습 vs 드리프트 테스트 (이동시킨 특성만 flag 기대) ===")
    print(ks_report(X_train, X_test_drift).to_string(index=False))

    # 4) 성능 비교 로그 포함
    preds_orig = predict_with_logging(X_test, model, y_true=y_test)
    preds_drift = predict_with_logging(X_test_drift, model, y_true=y_test)
    bal_orig = balanced_accuracy_score(y_test, preds_orig)
    bal_drift = balanced_accuracy_score(y_test, preds_drift)
    print("\n=== 성능 비교 (balanced accuracy) ===")
    print(f"원본 테스트셋    : {bal_orig:.3f}")
    print(f"드리프트 테스트셋: {bal_drift:.3f}")
    print(f"성능 저하        : {bal_orig - bal_drift:.3f}")

    # 5) 시계열 그래프: 드리프트가 커질수록 성능이 어떻게 변하는지
    plot_path = Path(__file__).resolve().parent.parent / "logs" / "drift_monitoring.png"
    drift_timeseries(model, X_test, y_test, plot_path)


if __name__ == "__main__":
    main()