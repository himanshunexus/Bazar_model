"""Shared inference logic: used by app.py (serving) and train.py (offline evaluation)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class Recommender:
    """Item-based collaborative filtering over a product x order matrix.

    model        : fitted sklearn NearestNeighbors (cosine, brute force) on the item matrix
    product_ids  : np.ndarray, row index -> real product_id
    item_matrix  : scipy CSR, shape (n_products, n_orders), binary
    popular_ids  : list[int], real product_ids sorted by popularity (cold-start fallback)
    """

    def __init__(self, model, product_ids, item_matrix, popular_ids, meta=None):
        self.model = model
        self.product_ids = np.asarray(product_ids)
        self.item_matrix = item_matrix
        self.popular_ids = [int(p) for p in popular_ids]
        self.meta = meta or {}
        self.pid_to_idx = {int(p): i for i, p in enumerate(self.product_ids)}

    @classmethod
    def load(cls, model_dir: str | Path) -> "Recommender":
        model_dir = Path(model_dir)
        model = joblib.load(model_dir / "model.pkl")
        bundle = joblib.load(model_dir / "product_ids.pkl")
        return cls(
            model=model,
            product_ids=bundle["product_ids"],
            item_matrix=bundle["item_matrix"],
            popular_ids=bundle["popular_ids"],
            meta={"trained_at": bundle.get("trained_at", "unknown")},
        )

    def recommend(self, recent_product_ids, top_n: int = 5) -> tuple[list[int], str]:
        """Return (recommended real product_ids, strategy label)."""
        recent = [int(p) for p in dict.fromkeys(recent_product_ids)]
        known = [self.pid_to_idx[p] for p in recent if p in self.pid_to_idx]
        seen = set(known)

        scores: dict[int, float] = {}
        if known:
            k = min(len(self.product_ids), top_n + len(known) + 25)
            distances, indices = self.model.kneighbors(self.item_matrix[known], n_neighbors=k)
            sims = 1.0 - distances
            for row_idx, row_sims in zip(indices, sims):
                for j, s in zip(row_idx, row_sims):
                    j = int(j)
                    if j in seen or s <= 0:
                        continue
                    scores[j] = scores.get(j, 0.0) + float(s)

        ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
        result = [int(self.product_ids[j]) for j in ranked]
        strategy = "collaborative" if result else "popular"

        # Pad with popular items so the caller always gets top_n (cold start / sparse neighbours)
        if len(result) < top_n:
            exclude = set(result) | set(recent)
            for pid in self.popular_ids:
                if pid not in exclude:
                    result.append(pid)
                    if len(result) == top_n:
                        break
            if ranked and len(result) > len(ranked):
                strategy = "collaborative+popular"
        return result, strategy
