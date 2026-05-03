# Toxic Comment Classifier

Multi-label toxicity classifier for social media comments using Bidirectional LSTM and Focal Loss.

## Overview

This project implements an end-to-end deep learning pipeline to automatically classify user-generated comments into six toxicity categories. The goal is to support content moderation at scale, reducing reliance on manual review.

The model is trained on ~160,000 English comments and produces a **binary vector of 6 elements** for each input comment - one value per toxicity category.

```
"You are an idiot!" → [1, 0, 0, 0, 1, 0]
                       ↑toxic       ↑insult
```

## Labels

| # | Label | Description |
|---|---|---|
| 1 | `toxic` | General toxic language |
| 2 | `severe_toxic` | Severely toxic content |
| 3 | `obscene` | Obscene language |
| 4 | `threat` | Threatening content |
| 5 | `insult` | Insulting language |
| 6 | `identity_hate` | Hate speech targeting identity |

## Architecture

- **Embedding** (128-dim, trained end-to-end) + SpatialDropout1D(0.3)
- **BiLSTM(128)** + BatchNormalization
- **BiLSTM(64)** + BatchNormalization
- **GlobalMaxPooling1D**
- **Dense(128)** + BatchNormalization + Dropout(0.4)
- **Dense(64)** + Dropout(0.3)
- **Dense(6, sigmoid)** - independent probability per label

**Key design choices:**
- **Focal Loss** (γ=2, α=0.25) instead of Weighted BCE to handle class imbalance without gradient explosion
- **Per-label threshold optimization** on the validation set to maximize F1 for each category independently
- **Multilabel stratified split** (70/15/15) to preserve class proportions across train/val/test

## Results

| Label | F1 | AUC |
|---|---|---|
| Toxic | 0.77 | 0.97 |
| Severely Toxic | 0.51 | 0.99 |
| Obscene | 0.82 | 0.99 |
| Threat | 0.15 | 0.96 |
| Insult | 0.70 | 0.98 |
| Identity Hate | 0.21 | 0.96 |

**Global metrics:** F1 Macro = 0.53 · F1 Micro = 0.70 · AUC-ROC (macro) = 0.97 · Hamming Loss = 0.023

The low F1 for `threat` and `identity_hate` reflects a structural data limitation (~0.3% positive rate), not an architectural one; both categories show AUC > 0.96, confirming the model has discriminative capacity that the F1 score alone does not capture.

## Dataset

~160,000 English comments, each labelled with zero or more of the six toxicity categories.

```
https://proai-datasets.s3.eu-west-3.amazonaws.com/Filter_Toxic_Comments_dataset.csv
```

## Notebook Structure

1. Setup and dependencies
2. EDA - raw data
3. Text preprocessing
4. Multilabel stratified split
5. Post-cleaning EDA and data-driven hyperparameter selection
6. Tokenization, padding and Focal Loss
7. Model architecture (Bidirectional LSTM)
8. Training - batch size search
9. Evaluation on test set
10. Inference
11. Summary and conclusions

## License

MIT
