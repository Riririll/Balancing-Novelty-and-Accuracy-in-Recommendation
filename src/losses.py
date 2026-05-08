"""
Loss functions (Paper Sections 3.1.3 - 3.1.5)
==============================================
- BPR loss for accuracy
- User-Genre contrastive loss for novelty (push users toward unexplored genres)
- Genre-Item contrastive loss for semantic consistency
- Combined multi-objective loss
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3.1.3  BPR
# ---------------------------------------------------------------------------
def bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    u: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> torch.Tensor:
    """
    Eq. 3.7:
        L_acc = - sum log sigmoid( s(u, i_pos) - s(u, i_neg) )
    """
    eu = user_emb[u]
    ep = item_emb[pos]
    en = item_emb[neg]
    s_pos = (eu * ep).sum(dim=-1)
    s_neg = (eu * en).sum(dim=-1)
    return -F.logsigmoid(s_pos - s_neg).mean()


# ---------------------------------------------------------------------------
# 3.1.4  User-Genre Contrastive  (Eq. 3.12)
# ---------------------------------------------------------------------------
def user_genre_contrastive(
    user_emb: torch.Tensor,
    genre_emb: torch.Tensor,
    user_indices: torch.Tensor,           # (B,)
    pos_genre_indices: torch.Tensor,      # (B,) one positive (unexplored) per user
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    InfoNCE loss pulling each user toward an *unexplored* genre while pushing
    away from all other genres in the catalog (used as in-batch negatives).

    For numerical stability we use F.cross_entropy on logits = sim(u, all_genres) / tau.
    """
    if user_indices.numel() == 0:
        return torch.tensor(0.0, device=user_emb.device, requires_grad=True)

    u = F.normalize(user_emb[user_indices], dim=-1)              # (B, D)
    g_all = F.normalize(genre_emb, dim=-1)                       # (G, D)

    logits = u @ g_all.t() / temperature                         # (B, G)
    return F.cross_entropy(logits, pos_genre_indices)


# ---------------------------------------------------------------------------
# 3.1.4  Genre-Item Contrastive  (Eq. 3.14)
# ---------------------------------------------------------------------------
def genre_item_contrastive(
    genre_emb: torch.Tensor,
    item_emb: torch.Tensor,
    genre_indices: torch.Tensor,          # (B,)
    pos_item_indices: torch.Tensor,       # (B,) item that belongs to that genre
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    InfoNCE pulling each genre embedding toward an item embedding of that genre,
    using all items as in-batch negatives.
    """
    if genre_indices.numel() == 0:
        return torch.tensor(0.0, device=genre_emb.device, requires_grad=True)

    g = F.normalize(genre_emb[genre_indices], dim=-1)            # (B, D)
    i_all = F.normalize(item_emb, dim=-1)                        # (N, D)

    logits = g @ i_all.t() / temperature                         # (B, N)
    return F.cross_entropy(logits, pos_item_indices)


# ---------------------------------------------------------------------------
# 3.1.5  Combined
# ---------------------------------------------------------------------------
def combined_loss(
    l_acc: torch.Tensor,
    l_uc: torch.Tensor,
    l_ci: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Paper uses L_nov = L_uc * L_ci  (Eq. 3.15) and  L = alpha*L_acc + beta*L_nov.
    Multiplying losses can be unstable when one is near zero, so we provide
    an additive variant that is easier to tune in practice.

    Set USE_PRODUCT_NOV = True below to follow the paper literally.
    """
    USE_PRODUCT_NOV = False
    l_nov = (l_uc * l_ci) if USE_PRODUCT_NOV else (l_uc + l_ci)
    return alpha * l_acc + beta * l_nov, l_nov
