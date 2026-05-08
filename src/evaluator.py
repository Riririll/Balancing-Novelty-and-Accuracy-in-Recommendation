"""
Evaluator (Paper Sections 4.1.3 - 4.1.4)
=========================================
Leave-One-Out evaluation with sampled-metrics protocol:
    - 1 positive (held-out test item) + 99 random negatives = 100 candidates
    - Rank, then compute metrics

Accuracy:  Recall@K, Precision@K, F1@K, NDCG@K, MRR
Novelty:   AVG Novelty@K, Unseen-Genre Ratio@K, Coverage@K, Long-tail Share@K
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .data_loader import TripartiteData


class LeaveOneOutEvaluator:
    """Implements the evaluation protocol from Section 4.1.3."""

    def __init__(
        self,
        td: TripartiteData,
        ks: List[int] = (5, 10, 20, 50),
        num_negatives: int = 99,
        long_tail_pct: float = 0.10,
        seed: int = 42,
    ):
        self.td = td
        self.ks = list(ks)
        self.num_negatives = num_negatives
        self.long_tail_pct = long_tail_pct
        self.rng = np.random.default_rng(seed)

        # Pre-compute long-tail threshold (Eq. 4.9)
        self.long_tail_threshold = np.quantile(
            td.item_popularity, long_tail_pct
        )

        # Pre-sample negatives once for stable comparisons across models
        self._cache_eval_candidates()

    # ------------------------------------------------------------------
    def _cache_eval_candidates(self):
        """For each user, pre-sample 99 negatives (items they never interacted with)."""
        self.eval_users: List[int] = []
        self.eval_pos: List[int] = []
        self.eval_negs: List[np.ndarray] = []

        for u, pos in self.td.test_user_item.items():
            seen = set(self.td.train_user_items.get(u, set()))
            seen.add(pos)
            available = np.setdiff1d(
                np.arange(self.td.n_items), np.array(list(seen)), assume_unique=False
            )
            if len(available) < self.num_negatives:
                negs = self.rng.choice(available, self.num_negatives, replace=True)
            else:
                negs = self.rng.choice(available, self.num_negatives, replace=False)
            self.eval_users.append(u)
            self.eval_pos.append(pos)
            self.eval_negs.append(negs.astype(np.int64))

    # ------------------------------------------------------------------
    def evaluate(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
    ) -> Dict[str, float]:
        """Returns metric dict.  user_emb (U,D), item_emb (N,D)."""
        ks = self.ks
        max_k = max(ks)

        # accuracy accumulators
        recall = {k: 0.0 for k in ks}
        precision = {k: 0.0 for k in ks}
        ndcg = {k: 0.0 for k in ks}
        mrr_total = 0.0
        n_users = len(self.eval_users)

        # novelty accumulators
        novelty = {k: 0.0 for k in ks}
        ugr = {k: 0.0 for k in ks}
        longtail = {k: 0.0 for k in ks}
        recommended_items = {k: set() for k in ks}

        # popularity normalization for novelty (Eq. 4.6)
        Z = float(self.td.item_popularity.sum())
        norm = np.log2(Z + self.td.n_items)

        for u, pos, negs in zip(self.eval_users, self.eval_pos, self.eval_negs):
            candidates = np.concatenate(([pos], negs))     # (100,)
            scores = user_emb[u] @ item_emb[candidates].T  # (100,)

            # rank descending
            order = np.argsort(-scores)
            ranked_items = candidates[order]               # (100,) actual item ids

            # position of the positive (rank 1 = top)
            pos_rank = int(np.where(ranked_items == pos)[0][0]) + 1

            # accuracy
            mrr_total += 1.0 / pos_rank
            for k in ks:
                top_k = ranked_items[:k]
                hit = int(pos in top_k)
                recall[k] += hit
                precision[k] += hit / k
                if hit:
                    ndcg[k] += np.log(2) / np.log(pos_rank + 1)

            # novelty / coverage / long-tail / unseen-genre
            unexplored = set(self.td.unexplored_genres(u).tolist())
            for k in ks:
                top_k = ranked_items[:k]
                # Novelty (Eq. 4.6)
                pop = self.td.item_popularity[top_k].astype(np.float64)
                p = (pop + 1.0) / (Z + self.td.n_items)
                novelty[k] += float(np.mean(-np.log2(p) / norm))

                # Unseen-Genre Ratio (Eq. 4.7)
                count_unseen = 0
                for it in top_k:
                    item_g = set(self.td.item_genres.get(int(it), []))
                    if item_g & unexplored:
                        count_unseen += 1
                ugr[k] += count_unseen / k

                # Long-tail share (Eq. 4.9)
                lt = (self.td.item_popularity[top_k] <= self.long_tail_threshold).sum()
                longtail[k] += lt / k

                # Coverage (Eq. 4.8)
                recommended_items[k].update(top_k.tolist())

        # ----- average over users
        results: Dict[str, float] = {}
        for k in ks:
            r = recall[k] / n_users
            p = precision[k] / n_users
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            results[f"Recall@{k}"] = r
            results[f"Precision@{k}"] = p
            results[f"F1@{k}"] = f1
            results[f"NDCG@{k}"] = ndcg[k] / n_users
            results[f"Novelty@{k}"] = novelty[k] / n_users
            results[f"UGR@{k}"] = ugr[k] / n_users
            results[f"LongTail@{k}"] = longtail[k] / n_users
            results[f"Coverage@{k}"] = len(recommended_items[k]) / self.td.n_items
        results["MRR"] = mrr_total / n_users

        return results
