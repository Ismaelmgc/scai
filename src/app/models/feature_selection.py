"""Feature selection – gain importance, mutual information, and correlation pruning.

Reduces feature count from ~150+ to top 30-50 to prevent overfitting,
especially important with small-cap data where row count is limited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer

from app.utils import get_logger

log = get_logger(__name__)


def _safe_impute(X: np.ndarray) -> np.ndarray:
    """Impute NaN/Inf with median."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)


def select_by_mutual_info(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task: str = "classification",
    top_k: int = 40,
    random_state: int = 42,
) -> list[str]:
    """Select top-k features by mutual information with the target.

    Parameters
    ----------
    df : DataFrame with features and label.
    feature_cols : Candidate feature columns.
    label_col : Target column.
    task : 'classification' or 'regression'.
    top_k : Number of features to keep.
    random_state : Seed for MI estimation.

    Returns
    -------
    Sorted list of selected feature names (best first).
    """
    usable = df.dropna(subset=[label_col])
    available = [c for c in feature_cols if c in usable.columns]
    if len(available) <= top_k:
        return available

    X = _safe_impute(usable[available].values)
    y = usable[label_col].values

    if task == "classification":
        mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    else:
        mi = mutual_info_regression(X, y, random_state=random_state, n_neighbors=5)

    mi_df = pd.DataFrame({"feature": available, "mi_score": mi})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    selected = mi_df.head(top_k)["feature"].tolist()
    log.info(
        "feature_selection_mi",
        total=len(available), selected=len(selected),
        top_mi=f"{mi_df.iloc[0]['mi_score']:.4f}",
        bottom_mi=f"{mi_df.iloc[min(top_k, len(mi_df))-1]['mi_score']:.4f}",
    )
    return selected


def prune_correlated(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.90,
) -> list[str]:
    """Remove highly correlated features, keeping the first in each pair.

    Parameters
    ----------
    df : DataFrame with features.
    feature_cols : Feature columns to check.
    threshold : Max absolute correlation between kept features.

    Returns
    -------
    Pruned list of feature names.
    """
    available = [c for c in feature_cols if c in df.columns]
    if len(available) <= 1:
        return available

    corr = df[available].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        highly_corr = upper.index[upper[col] > threshold].tolist()
        to_drop.update(highly_corr)

    kept = [c for c in available if c not in to_drop]
    if to_drop:
        log.info("feature_pruning_corr", dropped=len(to_drop), kept=len(kept),
                 threshold=threshold)
    return kept


def select_by_gain(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task: str = "classification",
    top_k: int = 40,
    random_state: int = 42,
) -> list[str]:
    """Select top-k features by LightGBM native feature importance (gain).

    Falls back to mutual information if LightGBM is unavailable.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lgb_unavailable_falling_back_to_mi")
        return select_by_mutual_info(df, feature_cols, label_col, task, top_k, random_state)

    usable = df.dropna(subset=[label_col])
    available = [c for c in feature_cols if c in usable.columns]
    if len(available) <= top_k:
        return available

    X = _safe_impute(usable[available].values)
    y = usable[label_col].values

    # Train a quick LightGBM model for importance
    if task == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=2.0,
            verbose=-1, random_state=random_state, n_jobs=1,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=15, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=2.0,
            verbose=-1, random_state=random_state, n_jobs=1,
        )

    # Use last 15% as eval, with 10-day purge gap to avoid autocorrelation leakage
    split = int(len(X) * 0.85)
    purge_gap = 10  # days to skip between train and eval
    train_end = max(split - purge_gap, int(len(X) * 0.5))
    model.fit(X[:train_end], y[:train_end])

    # Use native LightGBM feature importance (gain) instead of SHAP
    importances = model.feature_importances_

    imp_df = pd.DataFrame({"feature": available, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    selected = imp_df.head(top_k)["feature"].tolist()
    log.info(
        "feature_selection_gain",
        total=len(available), selected=len(selected),
        top_gain=f"{imp_df.iloc[0]['importance']:.6f}",
    )
    return selected


def select_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task: str = "classification",
    top_k: int = 40,
    corr_threshold: float = 0.85,
    random_state: int = 42,
    method: str = "shap",
) -> list[str]:
    """Full feature selection pipeline: prune correlation then rank.

    Parameters
    ----------
    method : 'gain' (default, LGB gain importance) or 'mi' (mutual information).

    Returns the top-k features after both filters.
    """
    # Step 1: prune highly correlated
    pruned = prune_correlated(df, feature_cols, threshold=corr_threshold)

    # Step 2: rank and keep top-k
    if method in ("shap", "gain"):
        selected = select_by_gain(
            df, pruned, label_col, task=task, top_k=top_k,
            random_state=random_state,
        )
        print(f"  ✓ Feature selection: {len(feature_cols)} → {len(selected)} features "
              f"(gain importance)", flush=True)
    else:
        selected = select_by_mutual_info(
            df, pruned, label_col, task=task, top_k=top_k,
            random_state=random_state,
        )

    log.info("feature_selection_complete",
             original=len(feature_cols), after_corr=len(pruned),
             final=len(selected), method=method)
    return selected
