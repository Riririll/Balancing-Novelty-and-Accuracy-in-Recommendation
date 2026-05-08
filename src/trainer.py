"""
Training loop for BaNANN (Paper Section 3.1.5)
================================================
Multi-objective optimization:
    L = alpha * L_acc(BPR) + beta * (L_uc + L_ci)

For each epoch:
  1. Build full graph forward pass (small dataset -> full-batch GAT)
  2. Sample BPR triples; compute BPR
  3. Sample (user, unexplored-genre) pairs; compute L_uc
  4. Sample (genre, item-in-genre) pairs; compute L_ci
  5. Combined backward + step
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from .data_loader import TripartiteData, BPRTripleSampler
from .losses import bpr_loss, user_genre_contrastive, genre_item_contrastive, combined_loss
from .model import TripartiteHeteroGAT


# ---------------------------------------------------------------------------
def sample_user_unexplored(
    td: TripartiteData,
    batch_size: int,
    rng: np.random.Generator,
):
    """Sample (user, unexplored_genre) pairs for L_uc."""
    users, genres = [], []
    attempts, max_attempts = 0, batch_size * 10
    while len(users) < batch_size and attempts < max_attempts:
        u = int(rng.integers(0, td.n_users))
        unexplored = td.unexplored_genres(u)
        if len(unexplored) == 0:
            attempts += 1
            continue
        g = int(rng.choice(unexplored))
        users.append(u)
        genres.append(g)
        attempts += 1
    if len(users) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(genres, dtype=torch.long),
    )


def sample_genre_item(
    td: TripartiteData,
    batch_size: int,
    rng: np.random.Generator,
):
    """Sample (genre, item-in-genre) pairs for L_ci."""
    # Pre-compute genre -> [items] once (cheap on small data)
    if not hasattr(sample_genre_item, "_cache"):
        cache: Dict[int, list] = {g: [] for g in range(td.n_genres)}
        for it, gs in td.item_genres.items():
            for g in gs:
                cache[g].append(it)
        sample_genre_item._cache = cache  # type: ignore[attr-defined]
    cache: Dict[int, list] = sample_genre_item._cache  # type: ignore[attr-defined]

    genres, items = [], []
    valid_genres = [g for g, lst in cache.items() if lst]
    if not valid_genres:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    for _ in range(batch_size):
        g = int(rng.choice(valid_genres))
        i = int(rng.choice(cache[g]))
        genres.append(g)
        items.append(i)
    return (
        torch.tensor(genres, dtype=torch.long),
        torch.tensor(items, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
class BaNANNTrainer:
    def __init__(
        self,
        model: TripartiteHeteroGAT,
        td: TripartiteData,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.td = td
        self.cfg = cfg
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"]["weight_decay"],
        )

        # Move graph to device
        self.edge_index_dict = {
            k: v.to(device) for k, v in td.hetero.edge_index_dict.items()
        }

        # Triple sampler for BPR
        self.bpr_sampler = BPRTripleSampler(
            td,
            num_negatives=cfg["train"]["num_negatives"],
            seed=cfg["train"]["seed"],
        )
        self.rng = np.random.default_rng(cfg["train"]["seed"])

    # ------------------------------------------------------------------
    def train_one_epoch(self, use_contrastive: bool = True) -> Dict[str, float]:
        self.model.train()
        cfg_t = self.cfg["train"]
        batch_size = cfg_t["batch_size"]
        num_batches = max(1, len(self.bpr_sampler) // batch_size)

        running = {"loss": 0.0, "bpr": 0.0, "uc": 0.0, "ci": 0.0}

        for _ in range(num_batches):
            self.optimizer.zero_grad()

            # ---- forward over the whole graph (small dataset, full-batch GNN)
            embs = self.model(self.edge_index_dict)
            ue, ie, ge = embs["user"], embs["item"], embs["genre"]

            # ---- BPR
            u, ip, in_ = self.bpr_sampler.sample_batch(batch_size)
            u, ip, in_ = u.to(self.device), ip.to(self.device), in_.to(self.device)
            l_acc = bpr_loss(ue, ie, u, ip, in_)

            if use_contrastive:
                # ---- User-Genre contrastive
                uc_users, uc_genres = sample_user_unexplored(
                    self.td, batch_size, self.rng
                )
                uc_users = uc_users.to(self.device)
                uc_genres = uc_genres.to(self.device)
                l_uc = user_genre_contrastive(
                    ue, ge, uc_users, uc_genres,
                    temperature=cfg_t["temperature"],
                )

                # ---- Genre-Item contrastive
                ci_genres, ci_items = sample_genre_item(
                    self.td, batch_size, self.rng
                )
                ci_genres = ci_genres.to(self.device)
                ci_items = ci_items.to(self.device)
                l_ci = genre_item_contrastive(
                    ge, ie, ci_genres, ci_items,
                    temperature=cfg_t["temperature"],
                )
            else:
                l_uc = torch.tensor(0.0, device=self.device)
                l_ci = torch.tensor(0.0, device=self.device)

            loss, _ = combined_loss(
                l_acc, l_uc, l_ci,
                alpha=cfg_t["alpha"],
                beta=cfg_t["beta"],
            )

            loss.backward()
            self.optimizer.step()

            running["loss"] += float(loss.item())
            running["bpr"] += float(l_acc.item())
            running["uc"] += float(l_uc.item()) if isinstance(l_uc, torch.Tensor) else 0.0
            running["ci"] += float(l_ci.item()) if isinstance(l_ci, torch.Tensor) else 0.0

        for k in running:
            running[k] /= num_batches
        return running

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_embeddings(self):
        self.model.eval()
        embs = self.model(self.edge_index_dict)
        return embs

    # ------------------------------------------------------------------
    def fit(self, evaluator=None, use_contrastive: bool = True):
        cfg_t = self.cfg["train"]
        history = []
        for epoch in tqdm(range(1, cfg_t["epochs"] + 1), desc="Epochs"):
            losses = self.train_one_epoch(use_contrastive=use_contrastive)
            log = {"epoch": epoch, **losses}
            if evaluator is not None and (epoch % 5 == 0 or epoch == cfg_t["epochs"]):
                embs = self.get_embeddings()
                metrics = evaluator.evaluate(
                    embs["user"].detach().cpu().numpy(),
                    embs["item"].detach().cpu().numpy(),
                )
                log.update({f"metric/{k}": v for k, v in metrics.items()})
                tqdm.write(
                    f"[ep {epoch:03d}] loss={losses['loss']:.4f} "
                    f"bpr={losses['bpr']:.4f} uc={losses['uc']:.4f} ci={losses['ci']:.4f} "
                    f"| Recall@10={metrics.get('Recall@10', 0):.4f} "
                    f"NDCG@10={metrics.get('NDCG@10', 0):.4f} "
                    f"Novelty@10={metrics.get('Novelty@10', 0):.4f}"
                )
            history.append(log)
        return history
