"""Django-side client for the BAZAR recommendation microservice.

Usage (in a view):
    from apps.core.ml_client import get_recommendations
    ids = get_recommendations(request.user.id, recent_ids, top_n=6)

settings.py:
    ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000/predict")
"""
from __future__ import annotations

import logging
from typing import Iterable

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

ML_TIMEOUT_SECONDS = 1.5          # applies to connect and to read separately
BREAKER_SECONDS = 30              # after a failure, skip the ML call for this long
TRENDING_CACHE_SECONDS = 300
_BREAKER_KEY = "ml_service_down"

_session = requests.Session()     # reuses TCP connections between calls


def _ml_url() -> str:
    return getattr(settings, "ML_SERVICE_URL", "http://localhost:8000/predict")


def get_trending_product_ids(top_n: int = 5) -> list[int]:
    """Fallback: best-selling active products. Cached so a down ML service adds no DB load.

    TODO: adjust the imports/field names below to match your actual models.
    """
    key = f"trending_products:{top_n}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from django.db.models import Count
        from apps.products.models import Product  # adjust to your app/model

        ids = list(
            Product.objects.filter(is_active=True)
            .annotate(sold=Count("orderitem"))           # adjust to your reverse relation
            .order_by("-sold", "-id")
            .values_list("id", flat=True)[:top_n]
        )
    except Exception:
        logger.exception("Trending fallback failed")
        return []
    cache.set(key, ids, TRENDING_CACHE_SECONDS)
    return ids


def get_recommendations(user_id: int | None, recent_product_ids: Iterable[int], top_n: int = 5) -> list[int]:
    """Return recommended product IDs. Never raises: falls back to trending items."""
    if cache.get(_BREAKER_KEY):                           # service recently down, don't pay the timeout again
        return get_trending_product_ids(top_n)

    payload = {
        "user_id": user_id,
        "recent_product_ids": [int(p) for p in recent_product_ids][:100],
        "top_n": top_n,
    }
    try:
        resp = _session.post(_ml_url(), json=payload, timeout=ML_TIMEOUT_SECONDS)
        resp.raise_for_status()
        ids = resp.json()["recommended_product_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise ValueError("Unexpected response shape from ML service")
        if ids:
            return ids[:top_n]
        logger.info("ML service returned no recommendations; using trending")
    except (requests.Timeout, requests.ConnectionError) as exc:
        logger.warning("ML service unreachable (%s); using trending fallback", exc.__class__.__name__)
        cache.set(_BREAKER_KEY, True, BREAKER_SECONDS)
    except (requests.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.error("ML service bad response: %s", exc)
        cache.set(_BREAKER_KEY, True, BREAKER_SECONDS)

    return get_trending_product_ids(top_n)
