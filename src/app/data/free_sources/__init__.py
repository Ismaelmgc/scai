"""Free data sources for SCAI historical extension and enrichment.

Architecture:
    Massive/Polygon = source of truth (2021+, vwap+transactions)
    Yahoo           = historical OHLCV extension (2019-2021 gap)
    SEC EDGAR       = fundamentals / corporate events (point-in-time)
    FRED            = macro/regime context

Usage:
    from app.data.free_sources.yahoo import download_yahoo_ohlcv
    from app.data.free_sources.sec_edgar import download_company_facts
    from app.data.free_sources.fred import download_fred_macro
"""
