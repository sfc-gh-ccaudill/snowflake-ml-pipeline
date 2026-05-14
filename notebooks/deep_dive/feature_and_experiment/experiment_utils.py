from dataclasses import dataclass
import os
import tempfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from snowflake.ml.dataset import load_dataset
from snowflake.snowpark import Session


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train_enc: np.ndarray
    y_test_enc: np.ndarray
    label_encoder: LabelEncoder
    preprocessor: ColumnTransformer
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    target_col: str


def prepare_data(
    session: Session,
    feature_config: dict[str, Any],
    db: str,
    schema: str,
    train_pdf: pd.DataFrame,
) -> PreparedData:
    train_pdf.columns = [c.upper() for c in train_pdf.columns]

    test_pdf = session.table(f"{db}.{schema}.TEST_FEATURES").to_pandas()
    test_pdf.columns = [c.upper() for c in test_pdf.columns]

    numeric_cols = feature_config["all_numeric_features"]
    categorical_cols = feature_config["all_categorical_features"]
    feature_cols = numeric_cols + categorical_cols
    target_col = feature_config["target_column"]

    X_train = train_pdf[feature_cols]
    y_train = train_pdf[target_col]
    X_test = test_pdf[feature_cols]
    y_test = test_pdf[target_col]

    le = LabelEncoder()
    le.fit(sorted(y_train.unique()))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Train: {len(train_pdf):,} rows | Test: {len(test_pdf):,} rows")
    print(
        f"Features: {len(feature_cols)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)"
    )
    print(f"Target: {target_col} -- classes: {list(le.classes_)}")
    print(f"Preprocessed shape: train={X_train_processed.shape}, test={X_test_processed.shape}")

    return PreparedData(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train_enc=y_train_enc,
        y_test_enc=y_test_enc,
        label_encoder=le,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
    )


def evaluate(model, X, y_true, label_encoder):
    y_pred = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_true)
    return {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "f1_weighted": float(
            f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
        ),
    }


def save_confusion_matrix(model, X, y_true, label_encoder, filepath):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = label_encoder.classes_
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(model, filepath, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f"feature_{i}" for i in indices])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(model, X, y_true, label_encoder, filepath):
    y_pred = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_true_labels = label_encoder.inverse_transform(y_true)
    report = classification_report(y_true_labels, y_pred_labels)
    with open(filepath, "w") as f:
        f.write(report)


def log_run_artifacts(exp, model, X_test, y_test_enc, label_encoder, artifact_dir, run_suffix):
    cm_path = os.path.join(artifact_dir, f"confusion_matrix_{run_suffix}.png")
    save_confusion_matrix(model, X_test, y_test_enc, label_encoder, cm_path)
    exp.log_artifact(cm_path, artifact_path="plots")

    fi_path = os.path.join(artifact_dir, f"feature_importance_{run_suffix}.png")
    save_feature_importance(model, fi_path)
    exp.log_artifact(fi_path, artifact_path="plots")

    cr_path = os.path.join(artifact_dir, f"classification_report_{run_suffix}.txt")
    save_classification_report(model, X_test, y_test_enc, label_encoder, cr_path)
    exp.log_artifact(cr_path, artifact_path="reports")


def get_model_params(model):
    import math

    return {
        k: v
        for k, v in model.get_params().items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }


from typing import Any

import numpy as np


def run_experiment(
    exp: Any,
    model: Any,
    experiment_name: str,
    train_features: np.array,
    train_labels: np.array,
    test_features: np.array,
    test_labels: np.array,
    le: LabelEncoder,
    artifact_dir: str = None,
):
    if artifact_dir is None:
        artifact_dir = tempfile.mkdtemp(prefix="exp_artifacts_")

    # == Log Params ==
    exp.log_params(get_model_params(model))

    # == Train Model ==
    model.fit(train_features, train_labels, eval_set=[(test_features, test_labels)], verbose=False)

    # == Evaluate Model ==
    metrics = evaluate(model, test_features, test_labels, le)
    exp.log_metrics(metrics)

    # == Log Experiment Artifacts ==
    log_run_artifacts(exp, model, test_features, test_labels, le, artifact_dir, experiment_name)

    # == Log step loss ==
    results = model.evals_result()
    for epoch, loss in enumerate(results["validation_0"]["mlogloss"]):
        exp.log_metric(key="validation_0:mlogloss", value=loss, step=epoch)

    print(f"{experiment_name} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nArtifacts logged: confusion_matrix, feature_importance, classification_report")

    return metrics
