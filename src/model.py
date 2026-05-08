"""
BaNANN model (Paper Section 3.1.2)
===================================
Heterogeneous Graph Attention Network over the tripartite user/item/genre graph.

We use PyG's GATConv applied per relation, wrapped in HeteroConv.
Each node type has its own learned embedding table (id-only features),
which is then refined through L message-passing layers.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class TripartiteHeteroGAT(nn.Module):
    """Hetero-GAT encoder for (user, item, genre) tripartite graph."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.embedding_dim = embedding_dim

        # --- ID embedding tables
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.genre_emb = nn.Embedding(n_genres, embedding_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.genre_emb.weight)

        # --- Hetero GAT layers (one HeteroConv per layer; one GATConv per edge type)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ("user", "rates", "item"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                    ("item", "ratesBy", "user"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                    ("user", "likes", "genre"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                    ("genre", "likesBy", "user"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                    ("genre", "has", "item"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                    ("item", "isIn", "genre"): GATConv(
                        embedding_dim, embedding_dim // num_heads,
                        heads=num_heads, dropout=dropout, add_self_loops=False,
                    ),
                },
                aggr="mean",
            )
            self.convs.append(conv)

        self.dropout = dropout

    # ------------------------------------------------------------------
    def forward(self, edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """Returns final embeddings dict: {'user': ..., 'item': ..., 'genre': ...}"""
        x_dict = {
            "user": self.user_emb.weight,
            "item": self.item_emb.weight,
            "genre": self.genre_emb.weight,
        }

        # Layer-wise message passing
        all_layers = [x_dict]
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                      for k, v in x_dict.items()}
            all_layers.append(x_dict)

        # Mean-pool across layers (LightGCN-style) for stability
        out = {}
        for k in ["user", "item", "genre"]:
            stacked = torch.stack([d[k] for d in all_layers], dim=0)
            out[k] = stacked.mean(dim=0)
        return out

    # ------------------------------------------------------------------
    def score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        u_idx: torch.Tensor,
        i_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 3.6 / 3.17:  score(u,i) = e_u^T e_i"""
        return (user_emb[u_idx] * item_emb[i_idx]).sum(dim=-1)
