"""
fraud_short_starter.py

Small fraud detection task:
- Tabular + Text
- Single sklearn pipeline + classifier
"""
import unittest
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)

class unit_test_prepare_X_y(unittest.TestCase):
    def test_prepare_X_y(self):
        # Expected number of columns: NUMERIC_COLS + CATEGORICAL_COLS + 1 TEXT_COL
        shape_x_col_exp = len(NUMERIC_COLS) + len(CATEGORICAL_COLS) + 1  # TEXT_COL is a single column
        # Expected number of rows: number of rows in the loaded data
        df = load_data()
        shape_x_row_exp = len(df)
        
        # Actual values from prepare_X_y (returns tuple: X, y)
        X, y = prepare_X_y(df)
        shape_x_row_act = len(X)
        shape_x_col_act = len(X.columns)
        
        print(f"Expected columns: {shape_x_col_exp}, Actual columns: {shape_x_col_act}")
        print(f"Expected rows: {shape_x_row_exp}, Actual rows: {shape_x_row_act}")
        
        # Assertions
        self.assertEqual(shape_x_col_exp, shape_x_col_act, 
                        f"Column count mismatch: expected {shape_x_col_exp}, got {shape_x_col_act}")
        self.assertEqual(shape_x_row_exp, shape_x_row_act,
                        f"Row count mismatch: expected {shape_x_row_exp}, got {shape_x_row_act}")




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
    # print(df)
    return df


def prepare_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[
        NUMERIC_COLS
        + CATEGORICAL_COLS
        + [TEXT_COL]
    ].copy()
    y = df[TARGET_COL].to_numpy(dtype=int)
    # print('X = ' + str(X))
    # print('y = ' + str(y))
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
    print(preprocessor)
    return preprocessor


def build_model() -> Pipeline:
    preprocessor = build_preprocessor()

    # clf = LogisticRegression(
    #     max_iter=1000,
    #     solver="lbfgs",
    # )

    clf = RandomForestClassifier(
        n_estimators = 100,
        criterion = 'entropy',
        class_weight = 'balanced',
        max_depth=5,  # Add depth limit
        min_samples_split=5,  # Add minimum samples
        random_state=42  # Add for reproducibility

        
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
    pr_auc = average_precision_score(y_true, y_proba)  # This is MAP (Mean Average Precision)

    y_pred = (y_proba >= 0.3).astype(int)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # This is MAR (Mean Average Recall)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(np.abs(tpr - fpr))

     # Diagnostic information
    print("\n=== Diagnostic Information ===")
    print(f"Number of positive predictions: {y_pred.sum()}")
    print(f"Number of actual positives: {y_true.sum()}")
    print(f"Max predicted probability: {y_proba.max():.4f}")
    print(f"Min predicted probability: {y_proba.min():.4f}")
    print(f"Mean predicted probability: {y_proba.mean():.4f}")
    print(f"Fraud rate in test set: {y_true.mean():.4f}")

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # This is MAR (Mean Average Recall)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = np.max(np.abs(tpr - fpr))

    print("\n=== Model Performance Metrics ===")
    print("ROC-AUC:", roc_auc)
    print("MAP (PR-AUC):", pr_auc)  # Mean Average Precision
    print("Precision:", precision)
    print("MAR (Recall):", recall)  # Mean Average Recall
    print("F1@0.5:", f1)
    print("KS:", ks)
    
    # If no positive predictions, suggest lower threshold
    if y_pred.sum() == 0:
        print("\n⚠️  WARNING: No positive predictions at threshold 0.5")
        print("   Try a lower threshold or check model training")
        # Try different thresholds
        for threshold in [0.3, 0.2, 0.1, 0.05]:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            if y_pred_thresh.sum() > 0:
                prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                print(f"   At threshold {threshold}: Precision={prec:.4f}, Recall={rec:.4f}, Predictions={y_pred_thresh.sum()}")


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
    # unittest.main(exit= False)
    main()

# load_data()
# prepare_X_y(load_data())
# build_preprocessor()