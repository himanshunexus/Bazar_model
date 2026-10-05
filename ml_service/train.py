"""Generate synthetic market-basket orders, train the recommender, export artifacts.

Outputs (in ./model by default, override with MODEL_DIR):
    model.pkl        fitted sklearn NearestNeighbors (cosine) over the product x order matrix
    product_ids.pkl  dict: product_ids (row -> real id), popular_ids, item_matrix, trained_at

Swap generate_orders() for a real query (user_id, product_id, order_id) to train on live data.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recommender import Recommender

SEED = 42


def generate_orders(n_users=2000, n_products=300, n_clusters=15, n_orders=25000, seed=SEED) -> pd.DataFrame:
    """Simulate baskets with real structure: users favour a few product 'themes'
    (e.g. breakfast, cleaning, snacks), so co-purchase patterns exist to be learned."""
    rng = np.random.default_rng(seed)
    product_ids = np.arange(1001, 1001 + n_products)            # realistic, non-contiguous-from-0 ids
    product_cluster = rng.integers(0, n_clusters, n_products)
    popularity = rng.pareto(1.5, n_products) + 1.0              # a few bestsellers, long tail
    user_primary = rng.integers(0, n_clusters, n_users)
    user_secondary = rng.integers(0, n_clusters, n_users)

    members = {c: np.where(product_cluster == c)[0] for c in range(n_clusters)}
    rows = []
    for order_id in range(1, n_orders + 1):
        u = int(rng.integers(0, n_users))
        r = rng.random()
        cluster = user_primary[u] if r < 0.70 else user_secondary[u] if r < 0.90 else rng.integers(0, n_clusters)
        size = int(np.clip(2 + rng.poisson(2), 2, 8))

        pool = members[int(cluster)]
        if len(pool) == 0:
            pool = np.arange(n_products)
        w = popularity[pool] / popularity[pool].sum()
        basket = set(rng.choice(pool, size=min(size, len(pool)), replace=False, p=w).tolist())
        if rng.random() < 0.15:                                  # impulse buy from anywhere
            basket.add(int(rng.integers(0, n_products)))
        for idx in basket:
            rows.append((order_id, u + 1, int(product_ids[idx])))

    return pd.DataFrame(rows, columns=["order_id", "user_id", "product_id"])


def fit(df: pd.DataFrame):
    """Build the product x order matrix and fit NearestNeighbors. Returns a Recommender."""
    df = df.drop_duplicates(["order_id", "product_id"])
    product_ids = np.sort(df["product_id"].unique())
    pid_to_row = {int(p): i for i, p in enumerate(product_ids)}
    order_codes, order_index = pd.factorize(df["order_id"])
    rows = df["product_id"].map(pid_to_row).to_numpy()

    item_matrix = csr_matrix(
        (np.ones(len(df), dtype=np.float32), (rows, order_codes)),
        shape=(len(product_ids), len(order_index)),
    )
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1).fit(item_matrix)

    counts = df["product_id"].value_counts()
    popular_ids = counts.index[:100].tolist()
    return Recommender(model, product_ids, item_matrix, popular_ids)


def evaluate(rec: Recommender, test_df: pd.DataFrame, k=10, seed=SEED, max_orders=2000) -> dict:
    """Hit-rate@k: hide one item from each held-out basket, see if we recommend it back."""
    rng = np.random.default_rng(seed)
    baskets = test_df.groupby("order_id")["product_id"].apply(list)
    baskets = baskets[baskets.map(len) >= 3].sample(frac=1.0, random_state=seed).head(max_orders)
    hits = pop_hits = 0
    for items in baskets:
        hidden = items[int(rng.integers(0, len(items)))]
        visible = [p for p in items if p != hidden]
        recs, _ = rec.recommend(visible, top_n=k)
        hits += hidden in recs
        pop_hits += hidden in [p for p in rec.popular_ids if p not in visible][:k]
    n = max(len(baskets), 1)
    return {"orders_evaluated": len(baskets), f"hit_rate@{k}": hits / n, f"popularity_baseline@{k}": pop_hits / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "model"))
    parser.add_argument("--save-csv", action="store_true", help="also write the synthetic orders to orders.csv")
    args = parser.parse_args()
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = generate_orders()
    print(f"Generated {df['order_id'].nunique()} orders, {len(df)} rows, "
          f"{df['user_id'].nunique()} users, {df['product_id'].nunique()} products")
    if args.save_csv:
        df.to_csv("orders.csv", index=False)

    # 1) honest offline check: train on 80% of orders, test on the other 20%
    order_ids = df["order_id"].unique()
    test_orders = set(np.random.default_rng(SEED).choice(order_ids, size=len(order_ids) // 5, replace=False))
    is_test = df["order_id"].isin(test_orders)
    print("Offline evaluation:", evaluate(fit(df[~is_test]), df[is_test]))

    # 2) final model on all data
    rec = fit(df)
    joblib.dump(rec.model, out / "model.pkl", compress=3)
    joblib.dump(
        {
            "product_ids": rec.product_ids,
            "popular_ids": rec.popular_ids,
            "item_matrix": rec.item_matrix,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        out / "product_ids.pkl",
        compress=3,
    )
    print(f"Saved model.pkl and product_ids.pkl to {out.resolve()}")


if __name__ == "__main__":
    main()
