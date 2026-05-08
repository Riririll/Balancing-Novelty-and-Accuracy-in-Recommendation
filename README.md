# BaNANN: Balancing Novelty and Accuracy in Recommendation

Implementation of **BaNANN** (Balancing Novelty and Accuracy with a Tripartite Graph Neural Network and Dual Contrastive Learning) — a recommender system framework that integrates accuracy and novelty objectives directly during training.

## Project Structure

```
BaNANN/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml              # Hyperparameters & paths
├── data/
│   └── (MovieLens dataset goes here)
├── scripts/
│   ├── download_data.py         # Download MovieLens 100K
│   └── run_experiment.sh        # End-to-end run
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Tripartite graph construction
│   ├── model.py                 # BaNANN model (Hetero-GAT)
│   ├── losses.py                # BPR + dual contrastive loss
│   ├── trainer.py               # Training loop
│   ├── evaluator.py             # Leave-One-Out + metrics
│   ├── baselines.py             # GAT+BPR, DSPA-RS, SBNA-L
│   └── utils.py                 # Helpers, seeding, config
├── outputs/                     # Logs, checkpoints, results
└── main.py                      # Entry point
```

## Setup

```bash
# 1. Create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies (CPU)
pip install torch
pip install torch-geometric
pip install pandas numpy scikit-learn pyyaml tqdm

# OR for CUDA, follow https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 3. Download dataset
python scripts/download_data.py

# 4. Run experiment
python main.py --config configs/config.yaml
```

## Components (matching the paper)

| Paper section | Code file |
|---|---|
| 3.1.1 Tripartite Graph Construction | `src/data_loader.py` |
| 3.1.2 Hetero-GNN Learning (GAT) | `src/model.py` |
| 3.1.3 BPR Learning | `src/losses.py` :: `bpr_loss` |
| 3.1.4 Two-Step Contrastive Learning | `src/losses.py` :: `user_genre_contrastive`, `genre_item_contrastive` |
| 3.1.5 Multi-Objective Optimization | `src/trainer.py` |
| 3.1.6 Inference | `src/evaluator.py` |
| 4.1.3 Leave-One-Out Evaluation | `src/evaluator.py` |
| 4.1.4 Metrics (Recall, NDCG, MRR, Novelty, UGR, Coverage, Long-tail) | `src/evaluator.py` |
| 4.1.2 Baselines (GAT+BPR, DSPA-RS, SBNA-L) | `src/baselines.py` |

## Reproducing Tables 4.1 and 4.2

```bash
python main.py --model banann   --config configs/config.yaml
python main.py --model gat_bpr  --config configs/config.yaml   # Ablation
python main.py --model dspa_rs  --config configs/config.yaml
python main.py --model sbna_l   --config configs/config.yaml
```

Results are written to `outputs/results.csv`.
