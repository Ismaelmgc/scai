"""Quick smoke test for all v2 improvements."""
import sys
sys.path.insert(0, "src")

# 1. Test all new modules import cleanly
from app.models.feature_selection import select_features, select_by_mutual_info, prune_correlated
print("OK feature_selection module")

from app.features.microstructure import compute_microstructure_features
print("OK microstructure module")

from app.features.sector import sic_to_sector, assign_sectors, compute_sector_features
print("OK sector module")

from app.models.tabular import TabularModel
print("OK tabular module (with stacking, calibration, tuning)")

from app.features.pipeline import build_feature_matrix
print("OK pipeline module (with sector + microstructure + risk-adjusted)")

# 2. Test sector mapping
assert sic_to_sector(6020) == "Financials"
assert sic_to_sector(2830) == "Healthcare"
assert sic_to_sector(3674) == "Technology"
assert sic_to_sector(4911) == "Utilities"
assert sic_to_sector(1311) == "Energy"
assert sic_to_sector(None) == "Unknown"
print("OK sector mapping")

# 3. TabularModel enhancement flags
m = TabularModel(horizon=5, task="classification")
assert m.use_feature_selection is True
assert m.use_hyperparam_tuning is True
assert m.use_stacking is True
assert m.use_calibration is True

m2 = TabularModel(horizon=5, task="regression")
assert m2.use_calibration is False  # calibration only for classification
print("OK TabularModel enhancement flags")

# 4. Test feature pipeline with synthetic data
import numpy as np
import pandas as pd

np.random.seed(42)
tickers = ["AAAA", "BBBB", "CCCC"]
dates = pd.bdate_range("2023-01-01", "2024-12-31")
rows = []
for t in tickers:
    base = 10 + np.random.rand() * 50
    for d in dates:
        change = np.random.randn() * 0.02
        base *= (1 + change)
        rows.append({
            "date": d,
            "ticker": t,
            "open": base * (1 + np.random.randn() * 0.005),
            "high": base * (1 + abs(np.random.randn()) * 0.01),
            "low": base * (1 - abs(np.random.randn()) * 0.01),
            "close": base,
            "volume": int(np.random.randint(100_000, 1_000_000)),
            "vwap": base * (1 + np.random.randn() * 0.002),
            "transactions": int(np.random.randint(500, 5000)),
        })

ohlcv = pd.DataFrame(rows)
universe = [
    {"ticker": "AAAA", "sic_code": "6020"},
    {"ticker": "BBBB", "sic_code": "2830"},
    {"ticker": "CCCC", "sic_code": "3674"},
]

features = build_feature_matrix(ohlcv, universe=universe, horizons=[5])
print(f"OK features built: {len(features)} rows x {len(features.columns)} cols")

# Check sector was assigned
assert "sector" in features.columns
assert set(features["sector"].unique()) == {"Financials", "Healthcare", "Technology"}
print("OK sectors assigned from SIC codes")

# Check risk-adjusted target exists
assert "fwd_ret_5d_risk_adj" in features.columns
assert "fwd_ret_5d_risk_adj_positive" in features.columns
print("OK risk-adjusted targets")

# Check microstructure features
micro_cols = [c for c in features.columns if "vwap_dev" in c or "cs_spread" in c or "obv" in c]
assert len(micro_cols) >= 3, f"Expected microstructure features, got: {micro_cols}"
print(f"OK microstructure features ({len(micro_cols)} cols)")

# Check sector features
sec_cols = [c for c in features.columns if "sector_ret" in c or "ret_vs_sector" in c or "sector_breadth" in c]
assert len(sec_cols) >= 3, f"Expected sector features, got: {sec_cols}"
print(f"OK sector features ({len(sec_cols)} cols)")

# 5. Test training with all enhancements
train_data = features.dropna(subset=["fwd_ret_5d_positive", "fwd_ret_5d"])
print(f"\nTraining on {len(train_data)} rows...")

cls = TabularModel(
    horizon=5, task="classification",
    use_feature_selection=True,
    feature_selection_top_k=20,
    use_hyperparam_tuning=True,
    use_stacking=True,
    use_calibration=True,
)
import warnings
warnings.filterwarnings("ignore")
metrics = cls.train(train_data)
print(f"OK classification trained - AUC: {metrics.get('val_auc', 0):.4f}, features: {metrics.get('n_features', 0)}")

# Test prediction
pred = cls.predict(train_data.head(10))
assert len(pred) == 10
assert all(0 <= p <= 1 for p in pred), "Classification predictions should be probabilities"
print(f"OK prediction works, range [{pred.min():.3f}, {pred.max():.3f}]")

# Test feature importance
fi = cls.feature_importance()
assert len(fi) > 0
print(f"OK feature importance ({len(fi)} features)")

# 6. Test feature selection standalone
from app.models.feature_selection import select_features
feat_cols = [c for c in train_data.select_dtypes(include=[np.number]).columns
             if not c.startswith("fwd_ret_") and c not in ("ticker", "date", "open", "high", "low", "close", "volume")]
selected = select_features(train_data, feat_cols, "fwd_ret_5d_positive", top_k=15)
assert len(selected) <= 15
print(f"OK feature selection: {len(feat_cols)} -> {len(selected)}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
