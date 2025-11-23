"""
fraud_short_starter.py

Small fraud detection task:
- Tabular + Text
- Single sklearn pipeline + classifier
"""

from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
)


DATA_PATH = "data/transactions_small.csv"

NUMERIC_COLS = [
    "amount",
    "device_risk_score",
    "user_credit_score",
    "hour_of_day",
    "past_7d_txn_count",
    "past_7d_avg_amount",
]

CATEGORICAL_COLS = [
    "merchant_category",
    "country",
]

TEXT_COL = "description"
TARGET_COL = "is_fraud"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[
        NUMERIC_COLS
        + CATEGORICAL_COLS
        + [TEXT_COL]
    ].copy()
    y = df[TARGET_COL].to_numpy(dtype=int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
            )),
            ("svd", TruncatedSVD(
                n_components=64,
                random_state=0,
            )),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("text", text_transformer, TEXT_COL),
        ],
    )

    return preprocessor


def build_model() -> Pipeline:
    preprocessor = build_preprocessor()

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", clf),
        ]
    )

    return pipe


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    y_pred = (y_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(np.abs(tpr - fpr))

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)
    print("F1@0.5:", f1)
    print("KS:", ks)


def main():
    df = load_data()
    X, y = prepare_X_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=0,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_proba)


if __name__ == "__main__":
    main()
