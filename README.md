# HR‑GNN‑Robust

Hierarchical Restructuring (HR) is a plug‑in framework that boosts GNN robustness under edge perturbations via a hierarchical Bayesian model.

## Highlights
- **Node‑level**: +3–21% accuracy under 90% edge drops  
- **Graph‑level**: up to +38% under 50% edge drops  

## Quick Start

```bash
conda create -n hr-gnn python=3.8 -y
conda activate hr-gnn
pip install -r requirements.txt

# 1. Download datasets & configs
bash scripts/run_exp/experiment.sh

# 2. Node‑level experiments
bash scripts/node_train.sh

# 3. Graph‑level experiments
bash scripts/graph_train.sh

```


Configs for both trainings live in scripts/run_exp/. Edit them there as needed.