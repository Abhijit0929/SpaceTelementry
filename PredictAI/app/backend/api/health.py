"""
health.py (API route)

Phase 4 endpoint: GET /api/health/overview - the main data source for the
dashboard's fleet-wide health cards, spacecraft comparison, and channel
health table.

Also exposes GET /api/health/{channel} as a convenience for a single
channel's health (same shape as one entry in the overview's channel list,
plus full prediction detail).
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.backend.services.health_service import (
    compute_channel_health,
    compute_health_overview,
)


router = APIRouter(
    prefix="/api/health",
    tags=["Health"]
)


@router.get("/overview")
def get_health_overview(
    limit: Optional[int] = Query(
        default=None,
        description="Optional cap on number of channels scored, for fast iteration during UI development. Omit for the real fleet-wide number."
    )
):
    try:
        return compute_health_overview(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{channel}")
def get_channel_health(channel: str):
    try:
        return compute_channel_health(channel)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
