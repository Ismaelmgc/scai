# Archived research scripts

Finished experiments, kept as a record of **how** each conclusion was reached.
They are **not maintained and many no longer run**: the 2026-10 cleanup removed
the modules only they used (`app.models`, `app.backtest`, `app.reporting`, the
sentiment/meta features, several Polygon endpoints). Read them; don't build on
them. The outcome of each line of research is summarised in `PROJECT.md`
("Research log").

| Path | What it was |
|---|---|
| `01_*` … `24_*` | V1→V4 small-cap research: baseline, sector enrichment, HP search, regime sizing, cooldown/filter/exit sweeps, feature batches, V4 report |
| `36_*` … `48_*` | Post-leak-fix (purged) re-validation: risk layer, capital deployment, purged baseline/sizing/breadth, 10-day horizon |
| `run_smallcap_pipeline.py` | The original single-script V2 pipeline (universe → download → features → classifier/regressor ensemble → backtest). Its 4 production helpers now live in `app.data.smallcap_ohlcv`, `app.data.smallcap_universe` and `app.features.lags` |
| `liquidcap_research/` | S&P 500 product screen: feature-family screen, horizon screen, combos, LSTM pilot |
| `midcap/` | Mid-cap product attempt (rejected: universe-selection artefact) |
| `_*.py`, `_cleanup.sh` | One-off helpers (PWA icons, seed fixes, comparisons) |
