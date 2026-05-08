"""
BaNANN main entry point.
========================
Usage:
    python main.py --model banann   --config configs/config.yaml
    python main.py --model gat_bpr  --config configs/config.yaml
    python main.py --model dspa_rs  --config configs/config.yaml
    python main.py --model sbna_l   --config configs/config.yaml
    python main.py --model all      --config configs/config.yaml
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch

from src.utils import set_seed, load_config, get_device, ensure_dir
from src.data_loader import build_tripartite
from src.model import TripartiteHeteroGAT
from src.trainer import BaNANNTrainer
from src.evaluator import LeaveOneOutEvaluator
from src.baselines import DSPARSReranker, SBNALReranker, evaluate_with_reranker


# ---------------------------------------------------------------------------
def build_model(td, cfg):
    return TripartiteHeteroGAT(
        n_users=td.n_users,
        n_items=td.n_items,
        n_genres=td.n_genres,
        embedding_dim=cfg["model"]["embedding_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )


# ---------------------------------------------------------------------------
def run_banann(td, evaluator, cfg, device, *, use_contrastive: bool):
    model = build_model(td, cfg)
    trainer = BaNANNTrainer(model, td, cfg, device)
    trainer.fit(evaluator=evaluator, use_contrastive=use_contrastive)

    embs = trainer.get_embeddings()
    ue = embs["user"].detach().cpu().numpy()
    ie = embs["item"].detach().cpu().numpy()
    return ue, ie, model


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model",
        choices=["banann", "gat_bpr", "dspa_rs", "sbna_l", "all"],
        default="banann",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["train"]["device"])
    out_dir = ensure_dir(cfg["output"]["output_dir"])

    print(f"[device] {device}")
    print("[data] building tripartite graph...")
    td = build_tripartite(
        cfg["data"]["data_dir"],
        min_user=cfg["data"]["min_user_interactions"],
        min_item=cfg["data"]["min_item_interactions"],
    )
    print(f"  users={td.n_users}, items={td.n_items}, genres={td.n_genres}")
    print(f"  train interactions={sum(len(s) for s in td.train_user_items.values())}")
    print(f"  test users with held-out item={len(td.test_user_item)}")

    evaluator = LeaveOneOutEvaluator(
        td,
        ks=cfg["eval"]["ks"],
        num_negatives=cfg["eval"]["num_negative_eval"],
        long_tail_pct=cfg["eval"]["long_tail_pct"],
        seed=cfg["train"]["seed"],
    )

    all_results = {}

    # -------------------------------------------------------------------
    # Train backbone (we need an accuracy-trained encoder for re-rankers)
    # -------------------------------------------------------------------
    backbone_ue, backbone_ie = None, None
    backbone_model = None

    def train_backbone():
        nonlocal backbone_ue, backbone_ie, backbone_model
        if backbone_ue is None:
            print("\n[train] GAT+BPR backbone (accuracy-only)...")
            t0 = time.time()
            ue, ie, m = run_banann(td, evaluator, cfg, device, use_contrastive=False)
            print(f"  done in {time.time() - t0:.1f}s")
            backbone_ue, backbone_ie, backbone_model = ue, ie, m

    # -------------------------------------------------------------------
    if args.model in ("banann", "all"):
        print("\n[train] BaNANN (full)...")
        t0 = time.time()
        ue, ie, _ = run_banann(td, evaluator, cfg, device, use_contrastive=True)
        print(f"  done in {time.time() - t0:.1f}s")
        all_results["BaNANN"] = evaluator.evaluate(ue, ie)

    if args.model in ("gat_bpr", "all"):
        train_backbone()
        all_results["GAT+BPR"] = evaluator.evaluate(backbone_ue, backbone_ie)

    if args.model in ("dspa_rs", "all"):
        train_backbone()
        rer = DSPARSReranker(td, alpha=0.3)
        all_results["DSPA-RS"] = evaluate_with_reranker(
            evaluator, backbone_ue, backbone_ie, rer
        )

    if args.model in ("sbna_l", "all"):
        train_backbone()
        rer = SBNALReranker(td, beta=0.4, min_occ=1)
        all_results["SBNA-L"] = evaluate_with_reranker(
            evaluator, backbone_ue, backbone_ie, rer
        )

    # -------------------------------------------------------------------
    # Print + save results
    # -------------------------------------------------------------------
    print("\n=================== RESULTS ===================")
    metric_keys = sorted(next(iter(all_results.values())).keys())
    print(f"{'Model':<12}", *(f"{k:>14}" for k in metric_keys))
    for name, m in all_results.items():
        print(f"{name:<12}", *(f"{m[k]:>14.4f}" for k in metric_keys))

    # CSV
    csv_path = out_dir / cfg["output"]["results_file"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model"] + metric_keys)
        for name, m in all_results.items():
            w.writerow([name] + [f"{m[k]:.6f}" for k in metric_keys])
    print(f"\n[saved] {csv_path}")

    # JSON
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
