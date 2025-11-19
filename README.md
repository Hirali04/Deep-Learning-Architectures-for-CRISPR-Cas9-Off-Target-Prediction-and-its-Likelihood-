# Deep Learning Architectures for CRISPR-Cas9 Off-Target Prediction and its Likelihood

This repository contains the implementation of two novel deep learning models for predicting CRISPR-Cas9 off-target effects: **CRISPR-DWA** (Deep Window Attention) and **CRISPR-PROB** (Probabilistic Classification).

![CRISPR Deep Learning Research Poster](./Graphical%20Abstract.jpeg)


## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)


## Overview

CRISPR-Cas9 technology has revolutionized genome editing, but off-target effects remain a significant concern for therapeutic applications. This project presents two complementary deep learning approaches to predict and assess the likelihood of off-target binding events:

1. **CRISPR-DWA**: A sequence-to-sequence model that predicts potential off-target sequences
2. **CRISPR-PROB**: A binary classifier that estimates the probability of off-target binding

Both models leverage multi-head attention mechanisms and deep neural networks to capture complex patterns in DNA sequences and improve prediction accuracy.

## Models

### CRISPR-DWA (Deep Window Attention)

- **Purpose**: Predicts off-target DNA sequences given an on-target sequence
- **Architecture**: Multi-head attention + Deep feedforward networks with dynamic windowing
- **Key Features**:
  - **Dynamic Windowing**: Novel 3-phase training approach with progressively focused attention windows
  - Embedding dimension: 35
  - Output: 23-nucleotide off-target sequences

### CRISPR-PROB (Probabilistic Classification)

- **Purpose**: Predicts the probability of off-target binding
- **Architecture**: Attention-based classifier with batch normalization
- **Key Features**:
  - Multi-head attention
  - Binary classification (binding/non-binding)

## Dataset

The models are trained on comprehensive CRISPR datasets including:

- **SITE-Seq**: Genome-wide off-target detection dataset
- **Kleinstiver**: 5 gRNA experimental validation dataset
- **CIRCLE-seq**: 6 gRNA active off-target dataset

**Data Processing**:

- DNA sequences encoded using one-hot representation
- Base mapping: A=0, T=1, G=2, C=3, N=4, -=5
- Sequence length: 23 nucleotides
- Train/test split with reserved validation sequences

## Requirements

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
openpyxl>=3.0.7
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/crispr-deep-learning.git
cd crispr-deep-learning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required datasets:

- `SITE-Seq_offTarget_wholeDataset.csv`
- `Kleinstiver_5gRNA_wholeDataset.csv`
- `CIRCLE-seq_6gRNA_active_offTargets.xlsx`

## Usage

### Training CRISPR-DWA Model

```python
# Open CRISPR-DWA.ipynb in Jupyter notebook
jupyter notebook CRISPR-DWA.ipynb
```

The notebook includes:

- Data preprocessing and encoding
- Model architecture definition
- **3-phase dynamic windowing training process**
- Sequence generation and evaluation

### Training CRISPR-PROB Model

```python
# Open CRISPR-PROB.ipynb in Jupyter notebook
jupyter notebook CRISPR-PROB.ipynb
```

The notebook includes:

- Binary classification setup
- Balanced sampling with bootstrapping
- Model training with cross-entropy loss
- Performance evaluation metrics

### Making Predictions

Both models can be used for inference on new sequences. Example usage:

```python
# For DWA model - predicting off-target sequences
predicted_sequence = model_dwa.predict(on_target_sequence)

# For PROB model - predicting binding probability
binding_probability = model_prob.predict(sequence_pair)
```

## Results

The models demonstrate strong performance on standard benchmarks:

- **CRISPR-DWA**: Accurate sequence-to-sequence prediction of off-targets
- **CRISPR-PROB**: High AUC and F1-scores for binary classification
- Both models show improved performance over baseline methods

Detailed results and performance metrics are available in the respective notebooks.


## Key Features

- **Dynamic Window Attention**: Novel progressive windowing approach for focused sequence analysis
- **Transfer Learning**: Pre-trained embeddings for nucleotide representations
- **Balanced Training**: Bootstrap sampling for class balance in PROB model
- **Comprehensive Evaluation**: Multiple datasets and metrics

## Acknowledgments

- CRISPR community for open datasets
- PyTorch team for the deep learning framework
- Research institutions supporting genome editing research


