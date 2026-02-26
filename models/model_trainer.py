from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass
class WalkForwardResult:
    oos_probs: pd.Series
    oos_pred: pd.Series
    oos_true: pd.Series
    fold_scores: List[Dict[str, float]]
    feature_importance: pd.Series


def make_target(close: pd.Series, horizon: int, threshold: float) -> pd.Series:
    fwd = close.shift(-horizon) / close - 1.0
    y = (fwd > threshold).astype("int8")
    return y


def walk_forward_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    n_splits: int = 6,
) -> WalkForwardResult:
    valid_mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid_mask], y.loc[valid_mask]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    probs = pd.Series(index=X.index, dtype="float32")
    preds = pd.Series(index=X.index, dtype="int8")
    fold_scores: List[Dict[str, float]] = []
    importances: List[pd.Series] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBClassifier(
            **params,
            tree_method="hist",
            n_jobs=2,
            early_stopping_rounds=30,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype("int8")

        probs.iloc[test_idx] = prob
        preds.iloc[test_idx] = pred
        fold_scores.append(
            {
                "fold": fold,
                "auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else 0.5,
                "acc": float(accuracy_score(y_test, pred)),
            }
        )
        importances.append(pd.Series(model.feature_importances_, index=X.columns))

    fi = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return WalkForwardResult(probs, preds, y, fold_scores, fi)


def top_features_by_importance(fi: pd.Series, n: int = 20) -> List[str]:
    return fi.head(n).index.tolist()
