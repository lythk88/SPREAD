# SPREAD: Test-time Diverse Reasoning by Riemannian Activation Steering

This repository contains the official implementation of the AAAI 2026 paper:

> **[AAAI 2026] Test-time Diverse Reasoning by Riemannian Activation Steering**

SPREAD is a test-time activation steering framework that induces **diverse reasoning trajectories** in large language models by performing steering on the **Riemannian manifold of hidden activations**, without any retraining or fine-tuning.

---

## Overview

- We propose the SPherical intervention for REAsoning Diversity (SPREAD), an unsupervised activation steering method that improves the diversity among reasoning trajectories. At a synchronization anchor, SPREAD extracts the hidden activations from all sequences, then computes the steering vectors that maximize the total
volume spanned by all possible subsets of the intervened
activations. SPREAD then adds these steering vectors to the respective activations of all subsequent tokens until the next synchronization anchor.
- We show that determining the optimal steering vectors
can be reformulated as a manifold optimization problem defined over the product of spheres, where the logdeterminant objective function captures the geometric diversity of the intervened activations. We propose using a Riemannian block coordinate descent algorithm, which exploits the product structure of the manifold constraints.
We also study the theoretical properties of the optimization problem and prove the convergence guarantee of the algorithm for appropriate step sizes. 

This repository provides code to reproduce the experiments reported in the paper.

---

## Repository Structure

```
SPREAD/main/
├── run_baseline.py        # Baseline inference methods
├── run_steering.py        # SPREAD: Riemannian activation steering
├── run_baseline.sh        # Shell script for baseline experiments
├── run_steering.sh        # Shell script for SPREAD experiments
└── README.md
```

---

## Environment Setup

We recommend using a Python virtual environment:

```bash
conda create -n spread python=3.10
conda activate spread
pip install -r requirements.txt
```

---

## Running Experiments

### Baseline Methods

```bash
bash main/run_baseline.sh
```

---

### SPREAD: Riemannian Activation Steering

```bash
bash main/run_steering.sh
```

---

## Citation

```bibtex
@inproceedings{spread2026,
  title     = {Test-time Diverse Reasoning by Riemannian Activation Steering},
  author    = {Ly Tran Ho Khanh and Dongxuan Zhu and Man-Chung Yue and Viet Anh Nguyen},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.08305},
}
```

---

## Contact

Please open a GitHub issue for questions or discussions.
