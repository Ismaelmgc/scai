"""Multi-model ensemble: LightGBM + XGBoost + CatBoost averaged.

Trains all three gradient boosting backends and combines predictions
via simple averaging (classification) or Ridge meta-learner (regression).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from app.config import get_settings
from app.utils import ensure_dir, get_logger

log = get_logger(__name__)

# --- optional imports ---
try:
    import lightgbm as lgb
    _HAS_LGB = True
except (ImportError, OSError):
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except (ImportError, OSError):
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CB = True
except (ImportError, OSError):
    _HAS_CB = False


class MultiModelEnsemble:
    """Train LightGBM + XGBoost + CatBoost and stack predictions."""

    def __init__(
        self,
        horizon: int = 5,
        task: Literal["classification", "regression"] = "classification",
        seed: int | None = None,
    ) -> None:
        self.horizon = horizon
        self.task = task
        self.seed = seed or get_settings().seed
        self.feature_cols: list[str] = []
        self._models: dict[str, Any] = {}
        self._meta: Any = None
        self._imputer = SimpleImputer(strategy="median")

    @property
    def _label_col(self) -> str:
        if self.task == "classification":
            return f"fwd_ret_{self.horizon}d_positive"
        return f"fwd_ret_{self.horizon}d"

    # ── Training ────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        val_frac: float = 0.15,
    ) -> dict[str, float]:
        label = self._label_col
        self.feature_cols = feature_cols

        usable = df.dropna(subset=[label])
        usable = usable.dropna(subset=self.feature_cols, how="all")
        usable = usable.sort_values("date")

        split_idx = int(len(usable) * (1 - val_frac))
        train_df = usable.iloc[:split_idx]
        val_df = usable.iloc[split_idx:]

        X_train = self._imputer.fit_transform(train_df[self.feature_cols].values)
        y_train = train_df[label].values
        X_val = self._imputer.transform(val_df[self.feature_cols].values)
        y_val = val_df[label].values

        # Train individual models
        oof_preds = {}  # out-of-fold predictions for meta-learner
        val_preds = {}

        if _HAS_LGB:
            oof, vp, m = self._train_lgb(X_train, y_train, X_val, y_val)
            self._models["lgb"] = m
            oof_preds["lgb"] = oof
            val_preds["lgb"] = vp
            log.info("lgb_trained")

        if _HAS_XGB:
            oof, vp, m = self._train_xgb(X_train, y_train, X_val, y_val)
            self._models["xgb"] = m
            oof_preds["xgb"] = oof
            val_preds["xgb"] = vp
            log.info("xgb_trained")

        if _HAS_CB:
            oof, vp, m = self._train_catboost(X_train, y_train, X_val, y_val)
            self._models["cb"] = m
            oof_preds["cb"] = oof
            val_preds["cb"] = vp
            log.info("catboost_trained")

        if not self._models:
            raise RuntimeError("No ML backends available (need lightgbm, xgboost, or catboost)")

        # Fit meta-learner for regression only (classification uses simple averaging)
        if self.task == "regression" and len(self._models) >= 2:
            self._fit_meta(X_train, y_train, X_val, y_val)

        # Compute validation metrics
        final_val_pred = self.predict_raw(X_val)
        metrics = self._compute_metrics(final_val_pred, y_val)
        metrics["n_models"] = len(self._models)
        metrics["models"] = ",".join(self._models.keys())
        metrics["n_features"] = len(self.feature_cols)
        metrics["n_train"] = len(train_df)
        metrics["n_val"] = len(val_df)
        log.info("multi_model_trained", **metrics)
        return metrics

    def _train_lgb(self, X_tr, y_tr, X_val, y_val):
        params = {
            "n_estimators": 1000, "learning_rate": 0.02, "max_depth": 4,
            "num_leaves": 15, "min_child_samples": 60,
            "subsample": 0.75, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 5.0,
            "random_state": self.seed, "n_jobs": 1, "verbose": -1,
        }
        if self.task == "classification":
            m = lgb.LGBMClassifier(**params)
        else:
            m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        oof = self._get_preds(m, X_tr)
        vp = self._get_preds(m, X_val)
        return oof, vp, m

    def _train_xgb(self, X_tr, y_tr, X_val, y_val):
        params = {
            "n_estimators": 1000, "learning_rate": 0.02, "max_depth": 4,
            "min_child_weight": 60, "subsample": 0.75, "colsample_bytree": 0.8,
            "reg_alpha": 0.3, "reg_lambda": 5.0, "gamma": 0.3,
            "random_state": self.seed, "n_jobs": 1, "verbosity": 0,
            "tree_method": "hist", "early_stopping_rounds": 80,
        }
        if self.task == "classification":
            params["eval_metric"] = "auc"
            m = xgb.XGBClassifier(**params)
        else:
            params["eval_metric"] = "rmse"
            m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof = self._get_preds(m, X_tr)
        vp = self._get_preds(m, X_val)
        return oof, vp, m

    def _train_catboost(self, X_tr, y_tr, X_val, y_val):
        params = {
            "iterations": 1000, "learning_rate": 0.02, "depth": 4,
            "l2_leaf_reg": 10.0, "random_seed": self.seed, "verbose": 0,
            "thread_count": 1, "allow_writing_files": False,
            "subsample": 0.75, "colsample_bylevel": 0.8,
            "bootstrap_type": "Bernoulli",
        }
        if self.task == "classification":
            m = CatBoostClassifier(**params)
        else:
            m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=80)
        oof = self._get_preds(m, X_tr)
        vp = self._get_preds(m, X_val)
        return oof, vp, m

    def _get_preds(self, model, X):
        if self.task == "classification" and hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def _fit_meta(self, X_train, y_train, X_val, y_val):
        """Fit Ridge meta-learner on CV-based out-of-fold predictions (regression only)."""
        n_splits = min(3, max(2, len(X_train) // 500))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_meta = np.zeros((len(X_train), len(self._models)))

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_f_tr, y_f_tr = X_train[tr_idx], y_train[tr_idx]
            X_f_val = X_train[val_idx]
            for col_idx, (name, model) in enumerate(self._models.items()):
                clone = self._clone_model(name)
                clone.fit(X_f_tr, y_f_tr)
                oof_meta[val_idx, col_idx] = self._get_preds(clone, X_f_val)

        mask = np.any(oof_meta != 0, axis=1)
        if mask.sum() < 50:
            self._meta = None
            return

        self._meta = Ridge(alpha=1.0)
        self._meta.fit(oof_meta[mask], y_train[mask])
        log.info("meta_learner_fitted", task=self.task, n_models=len(self._models))

    def _clone_model(self, name: str):
        """Create a fresh (unfitted) copy of a model by name."""
        if name == "lgb":
            params = {
                "n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
                "num_leaves": 15, "min_child_samples": 60,
                "subsample": 0.75, "colsample_bytree": 0.8,
                "reg_alpha": 0.3, "reg_lambda": 5.0,
                "random_state": self.seed, "n_jobs": 1, "verbose": -1,
            }
            if self.task == "classification":
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)
        elif name == "xgb":
            params = {
                "n_estimators": 400, "learning_rate": 0.03, "max_depth": 4,
                "min_child_weight": 60, "subsample": 0.75, "colsample_bytree": 0.8,
                "reg_alpha": 0.3, "reg_lambda": 5.0, "gamma": 0.3,
                "random_state": self.seed, "n_jobs": 1, "verbosity": 0,
                "tree_method": "hist",
            }
            if self.task == "classification":
                return xgb.XGBClassifier(**params)
            return xgb.XGBRegressor(**params)
        else:  # catboost
            params = {
                "iterations": 400, "learning_rate": 0.03, "depth": 4,
                "l2_leaf_reg": 10.0, "random_seed": self.seed,
                "verbose": 0, "thread_count": 1, "allow_writing_files": False,
                "bootstrap_type": "Bernoulli", "subsample": 0.75,
            }
            if self.task == "classification":
                return CatBoostClassifier(**params)
            return CatBoostRegressor(**params)

    # ── Predict ─────────────────────────────────────────────
    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble: Ridge meta-learner for regression, simple average for classification."""
        base_preds = np.column_stack([
            self._get_preds(m, X) for m in self._models.values()
        ])
        if self.task == "regression" and self._meta is not None:
            return self._meta.predict(base_preds)
        return base_preds.mean(axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._imputer.transform(df[self.feature_cols].values)
        return self.predict_raw(X)

    def predict_df(self, df: pd.DataFrame, col_name: str | None = None) -> pd.DataFrame:
        name = col_name or f"pred_{self.task}_{self.horizon}d"
        out = df[["ticker", "date"]].copy()
        out[name] = self.predict(df)
        return out

    def feature_importance(self) -> pd.DataFrame:
        """Average feature importance across all base models."""
        all_imp = []
        for name, m in self._models.items():
            if hasattr(m, "feature_importances_"):
                imp = m.feature_importances_
            elif hasattr(m, "get_feature_importance"):
                imp = m.get_feature_importance()
            else:
                continue
            if len(imp) == len(self.feature_cols):
                all_imp.append(imp / (imp.sum() + 1e-8))
        if not all_imp:
            return pd.DataFrame({"feature": self.feature_cols, "importance": 0, "importance_pct": 0})
        avg = np.mean(all_imp, axis=0)
        fi = pd.DataFrame({"feature": self.feature_cols, "importance": avg})
        fi = fi.sort_values("importance", ascending=False)
        total = fi["importance"].sum()
        fi["importance_pct"] = fi["importance"] / total if total > 0 else 0
        return fi

    def _compute_metrics(self, preds, y_val):
        metrics: dict[str, float] = {}
        if self.task == "classification":
            from sklearn.metrics import roc_auc_score, log_loss
            try:
                metrics["val_auc"] = float(roc_auc_score(y_val, preds))
                metrics["val_logloss"] = float(log_loss(y_val, np.clip(preds, 1e-7, 1 - 1e-7)))
            except ValueError:
                metrics["val_auc"] = 0.5
        else:
            from sklearn.metrics import mean_squared_error
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))
        return metrics

    # ── Serialization ───────────────────────────────────────
    def save(self, path: Path | str) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump({
                "models": self._models,
                "meta": self._meta,
                "imputer": self._imputer,
                "feature_cols": self.feature_cols,
                "horizon": self.horizon,
                "task": self.task,
            }, f)

    @classmethod
    def load(cls, path: Path | str) -> MultiModelEnsemble:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        obj = cls(horizon=data["horizon"], task=data["task"])
        obj._models = data["models"]
        obj._meta = data["meta"]
        obj._imputer = data["imputer"]
        obj.feature_cols = data["feature_cols"]
        return obj
