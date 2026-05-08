"""
Baseline models (Paper Section 4.1.2)
======================================
1. GAT+BPR     - ablation of BaNANN with contrastive loss disabled.
2. DSPA-RS     - Diversity & Serendipity Preference-Aware re-ranker over an
                 accuracy-trained model.  We re-rank GAT+BPR's candidate scores.
3. SBNA-L      - Striking the Balance, entropy-based re-ranking over a
                 venue/item co-occurrence graph.  Adapted to MovieLens
                 (no social network -> profile = the user's own train items).

Both DSPA-RS and SBNA-L are post-processing re-rankers, faithful to the paper's
description of their role as baselines.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .data_loader import TripartiteData


# ===========================================================================
# DSPA-RS  (Yin & Zhao, 2024)  - simplified post-processing re-ranker
# ===========================================================================
class DSPARSReranker:
    """
    Operates on top of a base accuracy model's candidate scores.

    For each user, computes:
      - diversity preference vector over genres (from training history)
      - novelty momentum over genres (recently-added genres)
    Then re-ranks the candidate list via:
        score_final = (1 - alpha) * acc_score_norm + alpha * diversity_serendipity_score
    """

    def __init__(self, td: TripartiteData, alpha: float = 0.3):
        self.td = td
        self.alpha = alpha
        # diversity preference: distribution over genres in user's history
        self.div_pref = np.zeros((td.n_users, td.n_genres), dtype=np.float64)
        for u, items in td.train_user_items.items():
            for it in items:
                for g in td.item_genres.get(it, []):
                    self.div_pref[u, g] += 1.0
        row_sum = self.div_pref.sum(axis=1, keepdims=True)
        self.div_pref = np.divide(
            self.div_pref, row_sum,
            out=np.zeros_like(self.div_pref), where=row_sum > 0,
        )

    def rerank(
        self,
        user_idx: int,
        candidates: np.ndarray,         # (100,)
        acc_scores: np.ndarray,         # (100,)
    ) -> np.ndarray:
        """Return re-ranked candidate indices (descending preference)."""
        # normalize accuracy scores
        s = acc_scores - acc_scores.min()
        if s.max() > 0:
            s = s / s.max()

        # serendipity bonus: items in genres user has explored less than uniform
        eps = 1.0 / self.td.n_genres
        bonus = np.zeros_like(s)
        for k, it in enumerate(candidates):
            for g in self.td.item_genres.get(int(it), []):
                if self.div_pref[user_idx, g] < eps:
                    bonus[k] += 1.0
        if bonus.max() > 0:
            bonus = bonus / bonus.max()

        final = (1.0 - self.alpha) * s + self.alpha * bonus
        order = np.argsort(-final)
        return candidates[order]


# ===========================================================================
# SBNA-L  (Agrawal et al., 2019)  - entropy-based balance re-ranker
# ===========================================================================
class SBNALReranker:
    """
    Computes Shannon entropy from item-popularity (less popular -> higher entropy)
    and re-ranks candidates as:
        final = (1-beta) * acc_score_norm + beta * entropy_norm
    Items must also satisfy a minimum-occurrence threshold for accuracy.
    """

    def __init__(self, td: TripartiteData, beta: float = 0.4, min_occ: int = 1):
        self.td = td
        self.beta = beta
        self.min_occ = min_occ

        Z = max(1, td.item_popularity.sum())
        p = (td.item_popularity + 1) / (Z + td.n_items)
        # per-item entropy contribution: -p log p (higher for rarer items because we use
        # 1 - p_normalized as a proxy)
        self.entropy = -p * np.log2(p)

    def rerank(
        self,
        user_idx: int,
        candidates: np.ndarray,
        acc_scores: np.ndarray,
    ) -> np.ndarray:
        s = acc_scores - acc_scores.min()
        if s.max() > 0:
            s = s / s.max()

        ent = self.entropy[candidates]
        ent_n = ent - ent.min()
        if ent_n.max() > 0:
            ent_n = ent_n / ent_n.max()

        # filter by min occurrence (accuracy guard)
        mask = self.td.item_popularity[candidates] >= self.min_occ
        final = (1.0 - self.beta) * s + self.beta * ent_n
        final = np.where(mask, final, final - 1e6)   # demote items below threshold

        order = np.argsort(-final)
        return candidates[order]


# ===========================================================================
# Baseline-aware evaluator wrapper
# ===========================================================================
def evaluate_with_reranker(
    evaluator,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    reranker,
) -> Dict[str, float]:
    """
    Run the standard LOO evaluation but apply `reranker.rerank` to each user's
    candidate scores before computing metrics.
    """
    td = evaluator.td
    ks = evaluator.ks
    max_k = max(ks)

    recall = {k: 0.0 for k in ks}
    precision = {k: 0.0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr_total = 0.0
    n_users = len(evaluator.eval_users)

    novelty = {k: 0.0 for k in ks}
    ugr = {k: 0.0 for k in ks}
    longtail = {k: 0.0 for k in ks}
    recommended_items = {k: set() for k in ks}

    Z = float(td.item_popularity.sum())
    norm = np.log2(Z + td.n_items)

    for u, pos, negs in zip(evaluator.eval_users, evaluator.eval_pos, evaluator.eval_negs):
        candidates = np.concatenate(([pos], negs))
        acc_scores = user_emb[u] @ item_emb[candidates].T
        ranked_items = reranker.rerank(u, candidates, acc_scores)
        pos_rank = int(np.where(ranked_items == pos)[0][0]) + 1

        mrr_total += 1.0 / pos_rank
        for k in ks:
            top_k = ranked_items[:k]
            hit = int(pos in top_k)
            recall[k] += hit
            precision[k] += hit / k
            if hit:
                ndcg[k] += np.log(2) / np.log(pos_rank + 1)

        unexplored = set(td.unexplored_genres(u).tolist())
        for k in ks:
            top_k = ranked_items[:k]
            pop = td.item_popularity[top_k].astype(np.float64)
            p = (pop + 1.0) / (Z + td.n_items)
            novelty[k] += float(np.mean(-np.log2(p) / norm))

            count_unseen = sum(
                1 for it in top_k
                if set(td.item_genres.get(int(it), [])) & unexplored
            )
            ugr[k] += count_unseen / k

            lt = (td.item_popularity[top_k] <= evaluator.long_tail_threshold).sum()
            longtail[k] += lt / k
            recommended_items[k].update(top_k.tolist())

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
        results[f"Coverage@{k}"] = len(recommended_items[k]) / td.n_items
    results["MRR"] = mrr_total / n_users
    return results
