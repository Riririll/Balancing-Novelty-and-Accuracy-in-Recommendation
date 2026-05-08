"""
Tripartite Graph Construction (Paper Section 3.1.1)
====================================================
Builds a heterogeneous graph with three node types:
    - user
    - item (movie)
    - genre

And six directed edge types (so message passing works in both directions):
    (user,  rates,    item),  (item, ratesBy, user)
    (user,  likes,    genre), (genre, likesBy, user)
    (genre, has,      item),  (item, isIn,    genre)

Implements Leave-One-Out splits (last interaction = test).
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Loading raw MovieLens
# ---------------------------------------------------------------------------
def load_movielens(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings_df, movies_df) from ml-latest-small."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings.csv")  # userId, movieId, rating, timestamp
    movies = pd.read_csv(data_dir / "movies.csv")    # movieId, title, genres ('Action|Comedy|...')
    return ratings, movies


def filter_kcore(
    ratings: pd.DataFrame,
    min_user: int = 5,
    min_item: int = 5,
) -> pd.DataFrame:
    """Drop users / items with too few interactions (k-core filtering)."""
    while True:
        u_counts = ratings.groupby("userId").size()
        i_counts = ratings.groupby("movieId").size()
        keep_u = u_counts[u_counts >= min_user].index
        keep_i = i_counts[i_counts >= min_item].index
        new = ratings[ratings.userId.isin(keep_u) & ratings.movieId.isin(keep_i)]
        if len(new) == len(ratings):
            return new
        ratings = new


# ---------------------------------------------------------------------------
# Indexing & splits
# ---------------------------------------------------------------------------
def build_index(values: pd.Series) -> Dict:
    """Map raw IDs to contiguous 0..N-1 ints."""
    return {v: i for i, v in enumerate(sorted(values.unique()))}


def leave_one_out_split(
    ratings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For each user, the last (latest timestamp) interaction goes to test."""
    ratings = ratings.sort_values(["userId", "timestamp"])
    test = ratings.groupby("userId").tail(1)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tripartite graph
# ---------------------------------------------------------------------------
class TripartiteData:
    """Container for the heterogeneous tripartite graph + bookkeeping."""

    def __init__(
        self,
        hetero: HeteroData,
        n_users: int,
        n_items: int,
        n_genres: int,
        user2idx: Dict,
        item2idx: Dict,
        genre2idx: Dict,
        train_user_items: Dict[int, set],         # u -> {item idx}
        test_user_item: Dict[int, int],           # u -> item idx (single)
        item_genres: Dict[int, List[int]],        # item idx -> [genre idx]
        item_popularity: np.ndarray,              # train interaction counts
        user_genre_counts: np.ndarray,            # (n_users, n_genres)
    ):
        self.hetero = hetero
        self.n_users = n_users
        self.n_items = n_items
        self.n_genres = n_genres
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.genre2idx = genre2idx
        self.train_user_items = train_user_items
        self.test_user_item = test_user_item
        self.item_genres = item_genres
        self.item_popularity = item_popularity
        self.user_genre_counts = user_genre_counts

    # Convenience: unexplored genres for user (Eq. 3.11)
    def unexplored_genres(self, user_idx: int) -> np.ndarray:
        counts = self.user_genre_counts[user_idx]
        total = counts.sum()
        if total == 0:
            return np.arange(self.n_genres)
        probs = counts / total
        eps = 1.0 / self.n_genres
        return np.where(probs < eps)[0]


def build_tripartite(
    data_dir: str | Path,
    min_user: int = 5,
    min_item: int = 5,
) -> TripartiteData:
    ratings, movies = load_movielens(data_dir)
    ratings = filter_kcore(ratings, min_user, min_item)

    # ------------- index users / items / genres
    user2idx = build_index(ratings["userId"])
    item2idx = build_index(ratings["movieId"])

    movies = movies[movies.movieId.isin(item2idx.keys())].copy()
    all_genres = set()
    for g in movies["genres"]:
        if isinstance(g, str) and g != "(no genres listed)":
            all_genres.update(g.split("|"))
    genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}

    n_users = len(user2idx)
    n_items = len(item2idx)
    n_genres = len(genre2idx)

    # ------------- map IDs and split
    ratings["u"] = ratings["userId"].map(user2idx)
    ratings["i"] = ratings["movieId"].map(item2idx)
    train_df, test_df = leave_one_out_split(ratings)

    # ------------- item -> genres
    item_genres: Dict[int, List[int]] = {}
    for _, row in movies.iterrows():
        idx = item2idx[row.movieId]
        gs = row.genres
        if isinstance(gs, str) and gs != "(no genres listed)":
            item_genres[idx] = [genre2idx[g] for g in gs.split("|") if g in genre2idx]
        else:
            item_genres[idx] = []
    # ensure every item is keyed
    for idx in range(n_items):
        item_genres.setdefault(idx, [])

    # ------------- training dictionaries
    train_user_items: Dict[int, set] = defaultdict(set)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        train_user_items[int(u)].add(int(i))

    test_user_item: Dict[int, int] = {
        int(u): int(i) for u, i in zip(test_df["u"].values, test_df["i"].values)
    }

    # ------------- popularity & user-genre counts
    item_popularity = np.zeros(n_items, dtype=np.int64)
    for _, i in zip(train_df["u"].values, train_df["i"].values):
        item_popularity[int(i)] += 1

    user_genre_counts = np.zeros((n_users, n_genres), dtype=np.int64)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        for g in item_genres[int(i)]:
            user_genre_counts[int(u), g] += 1

    # ------------- Build HeteroData (six directed edge types)
    hetero = HeteroData()

    # Node features: identity (we'll use learned embeddings instead)
    hetero["user"].num_nodes = n_users
    hetero["item"].num_nodes = n_items
    hetero["genre"].num_nodes = n_genres

    # user --rates--> item
    u_idx = train_df["u"].values.astype(np.int64)
    i_idx = train_df["i"].values.astype(np.int64)
    ui = torch.from_numpy(np.stack([u_idx, i_idx], axis=0))
    hetero["user", "rates", "item"].edge_index = ui
    hetero["item", "ratesBy", "user"].edge_index = ui.flip(0)

    # genre --has--> item
    gi_src, gi_dst = [], []
    for it, gs in item_genres.items():
        for g in gs:
            gi_src.append(g)
            gi_dst.append(it)
    gi = torch.tensor([gi_src, gi_dst], dtype=torch.long)
    hetero["genre", "has", "item"].edge_index = gi
    hetero["item", "isIn", "genre"].edge_index = gi.flip(0)

    # user --likes--> genre  (derived; weight = avg rating across genre items, Eq. 3.3)
    # For simplicity here, weight = count > 0; we just create the edges.
    ug_src, ug_dst = [], []
    for u in range(n_users):
        liked = np.where(user_genre_counts[u] > 0)[0]
        ug_src.extend([u] * len(liked))
        ug_dst.extend(liked.tolist())
    ug = torch.tensor([ug_src, ug_dst], dtype=torch.long)
    hetero["user", "likes", "genre"].edge_index = ug
    hetero["genre", "likesBy", "user"].edge_index = ug.flip(0)

    return TripartiteData(
        hetero=hetero,
        n_users=n_users,
        n_items=n_items,
        n_genres=n_genres,
        user2idx=user2idx,
        item2idx=item2idx,
        genre2idx=genre2idx,
        train_user_items=dict(train_user_items),
        test_user_item=test_user_item,
        item_genres=item_genres,
        item_popularity=item_popularity,
        user_genre_counts=user_genre_counts,
    )


# ---------------------------------------------------------------------------
# BPR pairwise sampler
# ---------------------------------------------------------------------------
class BPRTripleSampler:
    """Sample (u, i_pos, i_neg) triples for BPR training."""

    def __init__(self, td: TripartiteData, num_negatives: int = 1, seed: int = 42):
        self.td = td
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

        self.users: List[int] = []
        self.pos_items: List[int] = []
        for u, items in td.train_user_items.items():
            for i in items:
                self.users.append(u)
                self.pos_items.append(i)
        self.users = np.array(self.users, dtype=np.int64)
        self.pos_items = np.array(self.pos_items, dtype=np.int64)
        self.n_items = td.n_items

    def __len__(self):
        return len(self.users)

    def sample_batch(self, batch_size: int):
        idx = self.rng.integers(0, len(self.users), size=batch_size)
        u = self.users[idx]
        pos = self.pos_items[idx]
        neg = np.empty_like(pos)
        for k in range(batch_size):
            uu = int(u[k])
            seen = self.td.train_user_items[uu]
            j = int(self.rng.integers(0, self.n_items))
            while j in seen:
                j = int(self.rng.integers(0, self.n_items))
            neg[k] = j
        return (
            torch.from_numpy(u),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )
