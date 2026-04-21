# Experimental Results

## Evaluation Setting

- dataset: MovieLens 1M
- eval users: 300
- `K = 20`
- target size per user: 3
- ranking AUC uses held-out positives plus sampled negatives

## Recall Results

| Recall Method | Recall@20 | HitRate@20 | NDCG@20 |
|---|---:|---:|---:|
| ItemCF | 0.136667 | 0.320000 | 0.074504 |
| TF-IDF + FAISS | 0.037778 | 0.093333 | 0.020120 |
| Two-Tower | 0.040000 | 0.103333 | 0.017701 |

### Recall Takeaway

- ItemCF is the strongest recall method in the current setup
- TF-IDF + FAISS provides a useful content baseline
- Two-Tower completes the trainable recall route, but current lightweight features limit its performance

## Ranking Results

| Ranking Model | Recall@20 | HitRate@20 | NDCG@20 | AUC |
|---|---:|---:|---:|---:|
| MLP | 0.405556 | 0.713333 | 0.237271 | 0.659722 |
| DeepFM | 0.418889 | 0.723333 | 0.237815 | 0.670778 |

### Ranking Takeaway

- DeepFM improves over MLP across the current ranking metrics
- the ranking layer is stable and already suitable for a presentable recommendation-system baseline-plus-optimization narrative

## Final Summary

This project now supports the following comparison story:

- recall baselines:
  - ItemCF
  - TF-IDF + FAISS
- recall optimization:
  - Two-Tower
- ranking baseline:
  - MLP
- ranking optimization:
  - DeepFM

The current best-performing combination in this repository is:

- recall: ItemCF
- ranking: DeepFM
