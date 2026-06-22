"""
CardioCare 모델 학습
실행: python src/train.py
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from preprocessing import load_raw_data, clean_data, build_preprocessor, TARGET
from sklearn.model_selection import GridSearchCV
import joblib

RANDOM_STATE = 42


def make_pipeline(clf, scale=True):
    """대치 → (스케일링) → 특성선택 → 모델. 전부 train에만 fit → 누수 방지."""
    steps = [("preprocess", build_preprocessor())]
    if scale:                                    
        steps.append(("scale", StandardScaler()))
    steps.append(("select", SelectFromModel(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        threshold="median")))
    steps.append(("clf", clf))
    return Pipeline(steps)


def evaluate(y_true, y_pred):
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main():
    # 1) 로드 + 정리 + 분할
    df = clean_data(load_raw_data())
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"학습 {X_train.shape[0]}행 / 평가 {X_test.shape[0]}행\n")

    # 2) 비교할 모델 3종 — (분류기, 스케일링 필요 여부)
    models = {
        "logistic_regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "svc": (SVC(probability=True, random_state=RANDOM_STATE), True),
        "random_forest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE), False),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_tracking_uri((Path(__file__).resolve().parent.parent / "mlruns").as_uri())
    mlflow.set_experiment("CardioCare")

    results = []
    for name, (clf, scale) in models.items():
        pipe = make_pipeline(clf, scale=scale)

        # 5-fold 교차검증 (학습 데이터에서만)
        cv_bal_acc = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="balanced_accuracy"
        ).mean()

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = evaluate(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
            mask = pipe.named_steps["select"].get_support()
            selected = [f for f, m in zip(feat_names, mask) if m]

            # MLflow 기록: 태그 + 파라미터 + 지표 + 모델 아티팩트
            mlflow.set_tag("model_family", name)
            mlflow.log_param("scaling", scale)
            mlflow.log_param("n_features_selected", len(selected))
            mlflow.log_param("selected_features", ", ".join(selected))
            mlflow.log_metric("cv_balanced_accuracy", cv_bal_acc)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("false_negatives", int(cm[1, 0]))
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

            results.append({"model": name, "cv_bal_acc": cv_bal_acc,
                            **metrics, "FN": int(cm[1, 0])})
            print(f"[{name}] CV={cv_bal_acc:.3f} | test_bal_acc={metrics['balanced_accuracy']:.3f} "
                  f"recall={metrics['recall']:.3f} F1={metrics['f1']:.3f} FN={cm[1,0]}")

    # 3) 비교 표
    table = pd.DataFrame(results).set_index("model").round(3)
    print("\n=== 모델 비교 ===")
    print(table)

    # 4) 가장 유력한 후보(SVC)에 하이퍼파라미터 튜닝 (recall 기준)
    print("\n=== SVC 하이퍼파라미터 튜닝 (GridSearchCV) ===")
    svc_pipe = make_pipeline(SVC(probability=True, random_state=RANDOM_STATE), scale=True)
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01, 0.1],
        "clf__kernel": ["rbf"],
    }
    grid = GridSearchCV(svc_pipe, param_grid, cv=cv, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)
    print("최적 파라미터:", grid.best_params_)
    print(f"최적 CV recall: {grid.best_score_:.3f}")

    # 튜닝된 최종 모델 평가
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    final_metrics = evaluate(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== 튜닝된 SVC (테스트셋) ===")
    for k, v in final_metrics.items():
        print(f"{k:20s}: {v:.3f}")
    print("Confusion matrix [[TN FP],[FN TP]]:\n", cm)
    print(f"False Negatives: {cm[1, 0]}")

    # MLflow에 최종 모델 기록
    with mlflow.start_run(run_name="svc_tuned_final"):
        mlflow.set_tag("model_family", "svc")
        mlflow.set_tag("final_model", "true")
        mlflow.log_params(grid.best_params_)
        for k, v in final_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("false_negatives", int(cm[1, 0]))
        mlflow.sklearn.log_model(best_model, name="model", serialization_format="cloudpickle")

    # 최종 모델을 파일로 저장
    model_path = Path(__file__).resolve().parent.parent / "models" / "final_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\n최종 모델 저장 완료: {model_path}")


if __name__ == "__main__":
    main()