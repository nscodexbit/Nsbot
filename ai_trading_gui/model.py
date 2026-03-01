from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass
class ModelArtifacts:
    model: XGBClassifier
    best_params: dict
    train_proba: pd.Series
    test_proba: pd.Series
    feature_importance: pd.DataFrame
    metrics: dict
    confusion_matrix: np.ndarray
    test_index: pd.Index


def train_xgb_model(x: pd.DataFrame, y: pd.Series, train_size: float, random_state: int) -> ModelArtifacts:
    split_idx = int(len(x) * train_size)
    x_train, x_test = x.iloc[:split_idx], x.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_estimators=300,
    )

    param_dist = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    cv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)

    best: XGBClassifier = search.best_estimator_.set_params(early_stopping_rounds=25)
    val_cut = max(int(len(x_train) * 0.85), 1)
    x_fit, y_fit = x_train.iloc[:val_cut], y_train.iloc[:val_cut]
    x_val, y_val = x_train.iloc[val_cut:], y_train.iloc[val_cut:]
    eval_set = [(x_val, y_val)] if len(x_val) > 10 and y_val.nunique() > 1 else [(x_test, y_test)]
    best.fit(
        x_fit if len(x_val) > 10 else x_train,
        y_fit if len(x_val) > 10 else y_train,
        eval_set=eval_set,
        verbose=False,
    )

    train_proba = pd.Series(best.predict_proba(x_train)[:, 1], index=x_train.index)
    test_proba = pd.Series(best.predict_proba(x_test)[:, 1], index=x_test.index)

    pred = (test_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else 0.5,
    }
    conf = confusion_matrix(y_test, pred)

    fi = pd.DataFrame(
        {"feature": x.columns, "importance": best.feature_importances_}
    ).sort_values("importance", ascending=False)

    return ModelArtifacts(best, search.best_params_, train_proba, test_proba, fi, metrics, conf, x_test.index)


def walk_forward_validation(
    x: pd.DataFrame, y: pd.Series, train_size: float, random_state: int
) -> pd.DataFrame:
    split_idx = int(len(x) * train_size)
    step = max(len(x) // 10, 25)
    rows = []

    for i in range(split_idx, len(x), step):
        x_train, y_train = x.iloc[:i], y.iloc[:i]
        x_test = x.iloc[i : i + step]
        y_test = y.iloc[i : i + step]
        if len(x_test) < 5:
            continue

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        model.fit(x_train, y_train, verbose=False)
        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "start": x_test.index[0],
                "end": x_test.index[-1],
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.5,
            }
        )

    return pd.DataFrame(rows)
