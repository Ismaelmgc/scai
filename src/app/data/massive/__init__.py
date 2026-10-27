"""Massive.com (formerly Polygon.io) REST client — the small-cap data source.

- Reference data (ticker catalog incl. delisted, point-in-time details)
- Aggregates (daily bars, grouped daily)
- Snapshots (full-market)
- Corporate actions (splits, dividends)

``python -m app.data.massive`` runs the bulk ingestion jobs (``jobs/``).
"""

from app.data.massive.aggregates import AggregatesAPI
from app.data.massive.client import MassiveClient
from app.data.massive.corporate_actions import CorporateActionsAPI
from app.data.massive.reference import ReferenceAPI
from app.data.massive.snapshots import SnapshotsAPI

__all__ = [
    "MassiveClient",
    "ReferenceAPI",
    "AggregatesAPI",
    "SnapshotsAPI",
    "CorporateActionsAPI",
]
