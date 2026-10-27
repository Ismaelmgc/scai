"""Free data sources.

    Yahoo     = liquidcap prices (dividend-adjusted) + small-cap historical extension
    SEC EDGAR = fundamentals, point-in-time by filing date (both products)

Usage:
    from app.data.free_sources.yahoo import download_yahoo_ohlcv
    from app.data.free_sources.sec_edgar import download_company_facts
"""
