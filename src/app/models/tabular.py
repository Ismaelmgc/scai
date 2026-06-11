"""Tabular model for cross-sectional stock selection.

Uses LightGBM when available, falls back to sklearn GradientBoosting.
Supports hyperparameter tuning, feature selection, ensemble stacking,
and probability calibration.

Tasks:
  1. Classification – P(positive return) within the horizon.
  2. Regression    – expected return.
  3. Quantile      – p10, p50, p90 of the return distribution.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Try to import LightGBM; if unavailable, fall back to sklearn
try:
    import lightgbm as lgb
    _HAS_LGB = True
except (ImportError, OSError):
    _HAS_LGB = False

from app.config import get_settings
from app.utils import ensure_dir, get_logger

log = get_logger(__name__)


# ── Default feature columns (auto-detected if not given) ────
_EXCLUDE_COLS = frozenset({
    "ticker", "date", "open", "high", "low", "close", "volume",
    "sector", "exchange", "name", "is_active", "market_cap",
    "true_range", "dollar_volume", "obv",
    "sector_avg_volume",
})


def _auto_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding raw price / label columns."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_COLS
        and not c.startswith("fwd_")
        and not c.startswith("tb_label_")
        and not c.startswith("multi_horizon_")
    ]


# ── Hyperparameter search space ────────────────────────────
_HP_SPACE_CLS = {
    "model__n_estimators": [30, 50, 80, 120],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_samples_leaf": [20, 40, 60, 100],
    "model__subsample": [0.6, 0.7, 0.8, 0.9],
}

_HP_SPACE_REG = {
    "model__n_estimators": [30, 50, 80, 120],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_samples_leaf": [20, 40, 60, 100],
    "model__subsample": [0.6, 0.7, 0.8, 0.9],
}


class TabularModel:
    """Wrapper around LightGBM / sklearn for multi-task stock selection."""

    def __init__(
        self,
        horizon: int = 5,
        task: Literal["classification", "regression", "quantile"] = "classification",
        quantile_alpha: float = 0.5,
        lgb_params: dict[str, Any] | None = None,
        use_feature_selection: bool = True,
        feature_selection_top_k: int = 40,
        use_hyperparam_tuning: bool = True,
        use_stacking: bool = True,
        use_calibration: bool = True,
        use_walk_forward_cv: bool = True,
    ) -> None:
        self.horizon = horizon
        self.task = task
        self.quantile_alpha = quantile_alpha
        self._model: Any = None
        self._sklearn_model: Any = None
        self._use_lgb = _HAS_LGB
        self.feature_cols: list[str] = []

        # Enhancement flags
        self.use_feature_selection = use_feature_selection
        self.feature_selection_top_k = feature_selection_top_k
        self.use_hyperparam_tuning = use_hyperparam_tuning
        self.use_stacking = use_stacking
        self.use_calibration = use_calibration and task == "classification"
        self.use_walk_forward_cv = use_walk_forward_cv

        # Sensible defaults for small-cap data
        default_params: dict[str, Any] = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 7,
            "min_samples_leaf": 50,
            "subsample": 0.8,
            "random_state": get_settings().seed,
        }
        if lgb_params:
            default_params.update(lgb_params)
        self._params = default_params

    # ── Label helpers ───────────────────────────────────────
    @property
    def _label_col(self) -> str:
        if self.task == "classification":
            return f"fwd_ret_{self.horizon}d_positive"
        return f"fwd_ret_{self.horizon}d"

    # ── Feature Selection ───────────────────────────────────
    def _select_features(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> list[str]:
        """Apply MI-based feature selection if enabled."""
        if not self.use_feature_selection or len(feature_cols) <= self.feature_selection_top_k:
            return feature_cols

        from app.models.feature_selection import select_features
        return select_features(
            df, feature_cols, self._label_col,
            task=self.task,
            top_k=self.feature_selection_top_k,
            corr_threshold=0.90,
            random_state=self._params.get("random_state", 42),
        )

    # ── Train ───────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        val_frac: float = 0.15,
    ) -> dict[str, float]:
        """Train the model on a labelled feature matrix.

        The last ``val_frac`` of rows (by date) is used as a temporal
        validation set – **never** a random split.
        """
        label = self._label_col
        if label not in df.columns:
            raise ValueError(f"Label column '{label}' not in DataFrame")

        raw_features = feature_cols or _auto_feature_cols(df)
        self.feature_cols = self._select_features(df, raw_features)

        usable = df.dropna(subset=[label])
        usable = usable.dropna(subset=self.feature_cols, how="all")

        # Temporal split
        usable = usable.sort_values("date")
        split_idx = int(len(usable) * (1 - val_frac))
        train_df = usable.iloc[:split_idx]
        val_df = usable.iloc[split_idx:]

        X_train = train_df[self.feature_cols].values
        y_train = train_df[label].values
        X_val = val_df[self.feature_cols].values
        y_val = val_df[label].values

        if self._use_lgb:
            metrics = self._train_lgb(X_train, y_train, X_val, y_val)
        else:
            metrics = self._train_sklearn(X_train, y_train, X_val, y_val)

        metrics["n_train"] = len(train_df)
        metrics["n_val"] = len(val_df)
        metrics["n_features"] = len(self.feature_cols)
        metrics["backend"] = 1.0 if self._use_lgb else 0.0
        log.info("model_trained", task=self.task, horizon=self.horizon, **metrics)
        return metrics

    def _train_lgb(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict[str, float]:
        """Train with LightGBM."""
        params = {
            "n_estimators": self._params.get("n_estimators", 800),
            "learning_rate": self._params.get("learning_rate", 0.03),
            "max_depth": self._params.get("max_depth", 7),
            "num_leaves": self._params.get("num_leaves", 63),
            "min_child_samples": self._params.get("min_samples_leaf", 50),
            "subsample": self._params.get("subsample", 0.8),
            "colsample_bytree": self._params.get("colsample_bytree", 0.8),
            "reg_alpha": self._params.get("reg_alpha", 0.1),
            "reg_lambda": self._params.get("reg_lambda", 1.0),
            "random_state": self._params.get("random_state", 42),
            "n_jobs": 1,
            "verbose": -1,
        }
        if self.task == "classification":
            params["objective"] = "binary"
            model = lgb.LGBMClassifier(**params)
        elif self.task == "quantile":
            params["objective"] = "quantile"
            params["alpha"] = self.quantile_alpha
            model = lgb.LGBMRegressor(**params)
        else:
            params["objective"] = "regression"
            model = lgb.LGBMRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        self._model = model.booster_
        self._sklearn_model = model
        return self._compute_metrics(model, X_val, y_val)

    def _train_sklearn(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict[str, float]:
        """Train with sklearn — supports tuning, stacking, walk-forward CV, calibration."""
        log.info("using_sklearn_fallback", reason="LightGBM not available")

        common = {
            "n_estimators": min(self._params.get("n_estimators", 800), 80),
            "learning_rate": self._params.get("learning_rate", 0.05),
            "max_depth": min(self._params.get("max_depth", 7), 4),
            "min_samples_leaf": self._params.get("min_samples_leaf", 40),
            "subsample": self._params.get("subsample", 0.8),
            "random_state": self._params.get("random_state", 42),
        }

        if self.task == "classification":
            estimator = GradientBoostingClassifier(**common)
        elif self.task == "quantile":
            estimator = GradientBoostingRegressor(
                loss="quantile", alpha=self.quantile_alpha, **common
            )
        else:
            estimator = GradientBoostingRegressor(**common)

        base_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ])

        # ── Walk-forward CV for hyperparameter tuning ──────
        if self.use_hyperparam_tuning and self.task != "quantile" and len(X_train) >= 500:
            n_splits = min(3, max(2, len(X_train) // 300))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            hp_space = _HP_SPACE_CLS if self.task == "classification" else _HP_SPACE_REG
            scoring = "roc_auc" if self.task == "classification" else "neg_mean_squared_error"

            search = RandomizedSearchCV(
                base_pipe,
                param_distributions=hp_space,
                n_iter=15,
                cv=tscv,
                scoring=scoring,
                random_state=self._params.get("random_state", 42),
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            log.info("hyperparam_tuning_done",
                     best_params={k.replace("model__", ""): v
                                  for k, v in search.best_params_.items()},
                     best_score=f"{search.best_score_:.4f}")
        else:
            base_pipe.fit(X_train, y_train)
            best_pipe = base_pipe

        # ── Stacking ensemble ──────────────────────────────
        if self.use_stacking and self.task != "quantile" and len(X_train) >= 300:
            final_model = self._build_stacking(X_train, y_train, common)
        else:
            final_model = best_pipe

        # ── Calibration ────────────────────────────────────
        if self.use_calibration and self.task == "classification":
            final_model = self._calibrate_model(final_model, X_train, y_train)

        self._sklearn_model = final_model
        self._model = final_model
        return self._compute_metrics(final_model, X_val, y_val)

    def _build_stacking(
        self, X_train: np.ndarray, y_train: np.ndarray, common: dict,
    ) -> Any:
        """Build a stacking ensemble: GBM + RandomForest + Ridge/Logistic."""
        log.info("building_stacking_ensemble")
        seed = common.get("random_state", 42)

        if self.task == "classification":
            estimators = [
                ("gbm", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingClassifier(
                        n_estimators=common.get("n_estimators", 50),
                        learning_rate=common.get("learning_rate", 0.05),
                        max_depth=common.get("max_depth", 3),
                        min_samples_leaf=common.get("min_samples_leaf", 40),
                        subsample=common.get("subsample", 0.8),
                        random_state=seed,
                    )),
                ])),
                ("rf", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_leaf=30,
                        random_state=seed,
                        n_jobs=1,
                    )),
                ])),
            ]
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=500, random_state=seed),
                cv=3,
                passthrough=False,
                n_jobs=1,
            )
        else:
            estimators = [
                ("gbm", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(
                        n_estimators=common.get("n_estimators", 50),
                        learning_rate=common.get("learning_rate", 0.05),
                        max_depth=common.get("max_depth", 3),
                        min_samples_leaf=common.get("min_samples_leaf", 40),
                        subsample=common.get("subsample", 0.8),
                        random_state=seed,
                    )),
                ])),
                ("rf", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_leaf=30,
                        random_state=seed,
                        n_jobs=1,
                    )),
                ])),
            ]
            stack = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3,
                passthrough=False,
                n_jobs=1,
            )

        # Wrap in imputer pipeline
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X_train)
        stack.fit(X_imp, y_train)

        # Store imputer for prediction
        stack._scai_imputer = imputer
        log.info("stacking_ensemble_built", task=self.task)
        return stack

    def _calibrate_model(
        self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
    ) -> Any:
        """Apply isotonic calibration to classification model."""
        log.info("applying_calibration")
        try:
            cal = CalibratedClassifierCV(
                model,
                cv=TimeSeriesSplit(n_splits=2),
                method="isotonic",
            )
            cal.fit(X_train, y_train)
            log.info("calibration_applied", method="isotonic")
            return cal
        except Exception as e:
            log.warning("calibration_failed", error=str(e))
            return model

    def _compute_metrics(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.task == "classification":
            from sklearn.metrics import log_loss, roc_auc_score
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X_val)[:, 1]
            else:
                preds = model.predict(X_val)
            try:
                metrics["val_auc"] = float(roc_auc_score(y_val, preds))
                metrics["val_logloss"] = float(log_loss(y_val, preds))
            except ValueError:
                metrics["val_auc"] = 0.5
                metrics["val_logloss"] = 1.0
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            preds = model.predict(X_val)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, preds)))
            metrics["val_mae"] = float(mean_absolute_error(y_val, preds))
        return metrics

    # ── Predict ─────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw predictions (probabilities for classification)."""
        if self._model is None:
            raise RuntimeError("Model not trained – call .train() first")
        X = df[self.feature_cols].values

        model = self._sklearn_model or self._model

        # Handle stacking ensemble with custom imputer
        if hasattr(model, "_scai_imputer"):
            X = model._scai_imputer.transform(X)

        if self.task == "classification" and hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X)[:, 1])
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X))
        return np.asarray(self._model.predict(X))

    def predict_df(self, df: pd.DataFrame, col_name: str | None = None) -> pd.DataFrame:
        """Return df with a prediction column attached."""
        name = col_name or f"pred_{self.task}_{self.horizon}d"
        out = df[["ticker", "date"]].copy()
        out[name] = self.predict(df)
        return out

    # ── Feature importance ──────────────────────────────────
    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if self._model is None:
            return pd.DataFrame()

        model = self._model

        # Unwrap CalibratedClassifierCV
        if hasattr(model, "estimator"):
            model = model.estimator

        # LightGBM booster
        if _HAS_LGB and hasattr(model, "feature_importance"):
            imp = model.feature_importance(importance_type=importance_type)
        # Stacking model: average importances from base estimators
        elif hasattr(model, "estimators_") and hasattr(model, "final_estimator_"):
            imp = self._stacking_feature_importance(model)
        # sklearn Pipeline
        elif hasattr(model, "named_steps"):
            est = model.named_steps["model"]
            imp = getattr(est, "feature_importances_", np.zeros(len(self.feature_cols)))
        # sklearn estimator directly
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            imp = np.zeros(len(self.feature_cols))

        # Align lengths (sklearn pipeline may alter feature count)
        n_feats = len(self.feature_cols)
        n_imp = len(imp)
        if n_imp < n_feats:
            names = self.feature_cols[:n_imp]
        elif n_imp > n_feats:
            names = self.feature_cols + [f"feat_{i}" for i in range(n_feats, n_imp)]
        else:
            names = self.feature_cols
        fi = pd.DataFrame({
            "feature": names,
            "importance": imp[:len(names)],
        }).sort_values("importance", ascending=False)
        total = fi["importance"].sum()
        fi["importance_pct"] = fi["importance"] / total if total > 0 else 0
        return fi

    # ── Stacking feature importance ────────────────────────
    def _stacking_feature_importance(self, model: Any) -> np.ndarray:
        """Average feature importances from stacking base estimators."""
        all_imp = []
        for name, est in model.estimators_:
            if hasattr(est, "named_steps"):
                inner = est.named_steps.get("model", est)
            else:
                inner = est
            if hasattr(inner, "feature_importances_"):
                all_imp.append(inner.feature_importances_)
        if all_imp:
            # Pad to same length and average
            max_len = max(len(x) for x in all_imp)
            padded = [np.pad(x, (0, max_len - len(x))) for x in all_imp]
            return np.mean(padded, axis=0)
        return np.zeros(len(self.feature_cols))

    # ── Serialization ───────────────────────────────────────
    def save(self, path: Path | str) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "sklearn_model": self._sklearn_model,
                "feature_cols": self.feature_cols,
                "horizon": self.horizon,
                "task": self.task,
                "quantile_alpha": self.quantile_alpha,
                "use_lgb": self._use_lgb,
            }, f)
        log.info("model_saved", path=str(path))

    @classmethod
    def load(cls, path: Path | str) -> TabularModel:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        obj = cls(
            horizon=data["horizon"],
            task=data["task"],
            quantile_alpha=data.get("quantile_alpha", 0.5),
        )
        obj._model = data["model"]
        obj._sklearn_model = data.get("sklearn_model", data["model"])
        obj._use_lgb = data.get("use_lgb", _HAS_LGB)
        obj.feature_cols = data["feature_cols"]
        log.info("model_loaded", path=str(path))
        return obj
