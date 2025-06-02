
# Federated Transfer Learning for HPC Anomaly Detection

This repository contains the official implementation of experiments from the paper:  
**_Federated Transfer Learning for Anomaly Detection in Production-Scale HPC Systems_**, submitted to *Engineering Applications of Artificial Intelligence* (Elsevier, 2025).

## 🔍 Overview

High-Performance Computing (HPC) systems generate large-scale time-series data across thousands of compute nodes. Centralized anomaly detection methods face challenges related to scalability and data privacy. This project introduces a **Federated Transfer Learning (FTL)** framework using **dense autoencoders** trained with **FedAvg** to detect anomalies in a decentralized, privacy-preserving way.

### Key Features
- **Federated Learning with Dense Autoencoders**
- **FedAvg Aggregation Strategy**
- **Transfer Learning for Generalizing to Non-Participating Nodes**
- **Support for Supervised, Semi-Supervised, and Unsupervised Paradigms**
- **Top-N and Random-N Node Selection Strategies for Federated Participation**
- **Statistical Significance and Effect Size Reporting**

## 📁 Repository Structure

```
├── data/                   # Scripts for data preprocessing and formatting
├── models/                 # Dense autoencoder architecture and training code
├── federated/              # FedAvg-based federated learning implementation
├── experiments/            # Scripts for each experimental configuration
├── utils/                  # Metrics, data loading, and utility functions
├── results/                # Output logs and result summaries
├── notebooks/              # Optional analysis/visualization notebooks
└── README.md               # This file
```

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Running Experiments

To run a federated unsupervised experiment on dataset D1 using Top-10 nodes:

```bash
python experiments/run_federated_unsupervised.py --dataset D1 --nodes 10 --strategy topN
```

To apply transfer learning to the remaining nodes:

```bash
python experiments/run_transfer_learning.py --dataset D1 --pretrained_model path_to_model.pth
```

Scripts for supervised and semi-supervised settings are available in the `experiments/` directory.

## 📊 Results

Results include F1-score, AUC, ∆F1 gain after transfer learning, and statistical analyses.  
All experiments are conducted on real-world data from a production-scale HPC system (Marconi100 at CINECA).

## 📎 Citation

If you find this work useful, please cite:

```bibtex
@article{farooq2025ftl,
  title={Federated Transfer Learning for Anomaly Detection in Production-Scale HPC Systems},
  author={Farooq, Emmen and others},
  journal={Engineering Applications of Artificial Intelligence},
  year={2025}
}
```

## 🔗 Related Work

This work builds on our prior research:
- [Harnessing Federated Learning for Anomaly Detection in Supercomputer Nodes (FGCS 2024)](link)
- [A Federated Learning Approach for Anomaly Detection in HPC (ICTAI 2023)](link)

## 📬 Contact

For questions, collaboration, or dataset access, contact:  
📧 Emmen Farooq — *emmen.farooq3@unibo.it*
