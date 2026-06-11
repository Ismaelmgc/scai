# SCAI V4 — Final Validation Report

_Generated 2026-06-11 15:29. 16-fold walk-forward, 2022-06 → 2026-06, out-of-sample. Filter: price ≥ $1.50, ADV20 ≥ $500K, cost 15 bps/side unless noted._

| Config | WR (95% CI) | Trades | Monthly ret | Fold ret | Sharpe | Max DD | α vs mkt | α vs SPY | +Folds |
|---|---|---|---|---|---|---|---|---|---|
| **Old fiction (no filter, no cost)** | 49.4% (47%–52%) | 1,624 | +5.8% | +18.4% | +2.90 | -23.2% | +18.9% | +14.3% | 14/16 |
| **Honest baseline (filter + 15bps)** | 51.3% (49%–54%) | 1,624 | +4.1% | +12.7% | +2.52 | -22.1% | +13.2% | +8.6% | 14/16 |
| **V4 Baseline strategy (pt40)** | 51.4% (49%–54%) | 1,624 | +4.1% | +12.9% | +2.79 | -20.1% | +13.4% | +8.8% | 14/16 |
| **V4 Adaptive strategy** | 59.7% (57%–62%) | 1,624 | +3.6% | +11.1% | +2.73 | -15.1% | +11.6% | +7.0% | 14/16 |
| **Conservative bound (spread costs)** | 48.3% (46%–51%) | 1,624 | +2.8% | +8.6% | +1.42 | -25.0% | +9.1% | +4.6% | 12/16 |

## Honest reading

- **The old numbers were fiction.** The unfiltered backtest earned a large share of its return in stocks that could not actually be bought (sub-penny, $11–$153/day volume). Live trading proved it: 22 trades, all stopped out, portfolios −7% / +4% while the backtest promised +18%/fold.
- **What V4 actually changes**: tradability gate (selection only), 40% profit target on both strategies, adaptive tighten kept on the adaptive portfolio. Features and model unchanged — every candidate batch (rank transforms, microstructure, fundamentals) failed promotion criteria.
- **WR > 70% verdict**: not reachable without sacrificing returns. The honest envelope is ~51% (baseline) to ~60% (adaptive) with these CIs. Anyone promising 70%+ on 20-day small-cap holds is selling leakage.
- **Expected live performance**: between the deployed-strategy rows and the conservative bound. Paper-trade ≥ 30 closed trades before trusting any live number.

## Caveats

- 2021–2026 sample is mostly bull market; fold 2 (2022 bear) and fold 11 (2025 correction) are the realistic stress cases — both negative.
- Backtest fills at close with flat costs; real small-cap fills will be worse on thin names even post-filter.
- 16 folds ≈ 4 years. Fold-level means carry wide error bars themselves.