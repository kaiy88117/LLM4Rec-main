# LLM4Rec MovieLens 1M

This repository is a resume-ready recommendation-system project built on MovieLens 1M.

It is organized as a practical multi-stage recommender with:

- multiple recall baselines
- multiple ranking versions
- offline evaluation
- an online demo entry
- recommendation explanation

The project stops at a controlled scope on purpose:

- no extra ranking families beyond `MLP` and `DeepFM`
- no API service or frontend system
- no additional large engineering modules

## Project goal

The goal of this project is to turn a local runnable recommendation demo into a presentable recommendation-system project that can be explained clearly in interviews.

Core idea:

1. build a stable MovieLens 1M recommendation pipeline
2. separate the system into recall, ranking, explanation, and serving entry
3. compare baseline and optimized approaches with offline metrics
4. keep the code small enough to reproduce locally

Related reading:

- [Project Overview](docs/project_overview.md)
- [Experimental Results](docs/results.md)
- [Interview Notes](docs/interview_notes.md)

## Final architecture

```text
User Query / User History
        |
        v
  -------------------------
  Recall Layer
  - ItemCF baseline
  - TF-IDF + FAISS baseline
  - Two-Tower optimized recall
  -------------------------
        |
        v
  Candidate Movies
        |
        v
  -------------------------
  Ranking Layer
  - MLP baseline
  - DeepFM optimized ranking
  -------------------------
        |
        v
  Top-K Results
        |
        v
  Explanation Layer
  - template explanation
  - optional LLM explanation
        |
        v
  Online Demo
  - Recall_and_Rank.py
  - run_chat.py
```

## Project structure

```text
LLM4Rec-main/
|- agents/
|  |- recommender_agent.py
|- tools/
|  |- recommend_tool.py
|  |- recall_baselines.py
|  |- two_tower_recall.py
|  |- rec_explainer.py
|- scripts/
|  |- prepare_movielens.py
|  |- build_ctr_samples.py
|  |- train_ctr.py
|  |- train_deepfm.py
|  |- train_two_tower.py
|  |- evaluate_offline.py
|- data/
|  |- ml-1m/
|  |- processed/
|- models/
|- project_config.py
|- ranking_models.py
|- run_chat.py
`- Recall_and_Rank.py
```

## Raw data location

Raw MovieLens 1M data is expected under `data/ml-1m/`.

Supported local layouts:

1. `data/ml-1m/movies.dat`
2. `data/ml-1m/ml-1m/movies.dat`

Required files:

- `movies.dat`
- `ratings.dat`
- `users.dat`

## Environment setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Main runtime dependencies:

- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `joblib`
- `faiss-cpu`

Device rule:

- if `torch.cuda.is_available()` is `true`, training and ranking inference use CUDA
- otherwise they use CPU

## Baseline and optimized routes

### Recall

Baselines:

- `ItemCF`
  - collaborative recall
  - uses positive user-item co-occurrence
- `TF-IDF + FAISS`
  - content recall
  - uses movie title and genre text

Optimized recall:

- `Two-Tower`
  - learns separate user and item embeddings
  - retrieves items by embedding similarity
  - added as the recall optimization route in the final stage

### Ranking

Baseline:

- `MLP`
  - simple PyTorch ranker
  - input is user-genre vector + item-genre vector

Optimized ranking:

- `DeepFM`
  - adds linear term, FM interaction term, and deep nonlinear layers
  - better captures feature interactions on the same base features

## Build and run

### 1. Build processed artifacts and train the main models

```bash
python -m scripts.prepare_movielens
```

Repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_movielens
```

Optional parameters:

```bash
python -m scripts.prepare_movielens --max-positive-per-user 20 --negatives-per-positive 2 --epochs 5
```

This step builds:

- movie catalog
- movie stats
- user behavior summary
- ItemCF neighbors
- TF-IDF embeddings and FAISS index
- CTR samples
- MLP ranker
- Two-Tower recall model and item embeddings

### 2. Train DeepFM

```bash
python -m scripts.train_deepfm --epochs 5
```

Repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_deepfm --epochs 5
```

### 3. Run offline evaluation

```bash
python -m scripts.evaluate_offline --max-users 300 --k 20
```

Repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_offline --max-users 300 --k 20
```

### 4. Run the online demo

```bash
python Recall_and_Rank.py
python run_chat.py
```

Repository virtual environment:

```powershell
.\.venv\Scripts\python.exe Recall_and_Rank.py
.\.venv\Scripts\python.exe run_chat.py
```

## Metric definitions

- `Recall@K`
  - fraction of held-out positive targets retrieved in top K
- `HitRate@K`
  - whether at least one held-out positive target appears in top K
- `NDCG@K`
  - ranking-aware metric rewarding earlier correct hits
- `AUC`
  - binary ranking discrimination between positives and sampled negatives

## Experimental results

Evaluation setting:

- `eval_users = 300`
- `K = 20`
- `target_size = 3`
- ranking AUC uses held-out positives plus sampled negatives

### Recall comparison

| Recall Method | Recall@20 | HitRate@20 | NDCG@20 |
|---|---:|---:|---:|
| ItemCF | 0.136667 | 0.320000 | 0.074504 |
| TF-IDF + FAISS | 0.037778 | 0.093333 | 0.020120 |
| Two-Tower | 0.040000 | 0.103333 | 0.017701 |

### Ranking comparison

| Ranking Model | Recall@20 | HitRate@20 | NDCG@20 | AUC |
|---|---:|---:|---:|---:|
| MLP | 0.405556 | 0.713333 | 0.237271 | 0.659722 |
| DeepFM | 0.418889 | 0.723333 | 0.237815 | 0.670778 |

## Current interpretation

What the current experiments show:

- `ItemCF` is the strongest recall baseline on MovieLens 1M in the current setup
- `TF-IDF + FAISS` is weaker but provides a clean content-based baseline
- `Two-Tower` has been integrated successfully as a trainable recall optimization route, but in the current simple feature setup it does not yet beat ItemCF
- `DeepFM` consistently improves on the `MLP` ranking baseline

This is still a useful result:

- not every optimization beats a strong collaborative baseline immediately
- the project now demonstrates a real experimental comparison instead of only adding models

## Current limitations

- recall features are still relatively simple
- Two-Tower currently uses genre-level feature inputs, not richer sequence or ID embeddings
- content recall relies on TF-IDF rather than stronger semantic encoders
- ranking features are still lightweight compared with industrial recommenders
- the online demo is a CLI entry, not a full service

## Future directions

These are intentionally not implemented in this repository version:

- richer Two-Tower features using user/item IDs, histories, and metadata
- better semantic content recall encoders
- stronger sequential ranking features
- stricter experimental tracking and hyperparameter sweeps
- service/API deployment

## Friendly error handling

Raw data missing:

- raises `MissingRawDataError`
- points to `data/ml-1m/` or `data/ml-1m/ml-1m/`

Artifacts missing:

- raises `MissingArtifactError`
- tells you to rerun:

```bash
python -m scripts.prepare_movielens
```

## Interview talk track

### 1-minute version

I built a MovieLens 1M recommendation project with a full recall-ranking-explanation pipeline. On the recall side I implemented ItemCF and TF-IDF+FAISS as baselines, then added a trainable Two-Tower model as the recall optimization route. On the ranking side I used a PyTorch MLP baseline and a DeepFM optimization version. I also built offline evaluation with Recall@K, HitRate@K, NDCG@K, and AUC, then compared the methods quantitatively. The final result is a reproducible local project that is small enough to run but structured like a real recommendation system.

### 3-minute version

1. I first built a stable local MovieLens 1M pipeline with preprocessing, recall, ranking, explanation, and a CLI demo.
2. I split the recommendation problem into two major stages: recall and ranking.
3. For recall, I added two baselines:
   - ItemCF for collaborative filtering
   - TF-IDF + FAISS for content retrieval
4. Then I added a Two-Tower model as the recall optimization version, keeping the rest of the system unchanged so the comparison stayed fair.
5. For ranking, I kept a simple MLP baseline and added DeepFM as the optimized version.
6. I built an offline evaluation script to compare Recall@K, HitRate@K, NDCG@K, and AUC.
7. The experiments showed that ItemCF is still the strongest recall route on this dataset, while DeepFM improves over MLP on ranking. That gave me a concrete way to explain why collaborative signals matter on MovieLens and why feature interaction modeling helps ranking.

### What to emphasize in interviews

- the project is modular: recall, ranking, explanation, and serving are separated
- baselines and optimized versions are both implemented
- the model choices are backed by offline metrics
- the project shows both engineering closure and experimental reasoning

## Resume-ready summary

You can describe this project as:

- Built a MovieLens 1M recommendation system with modular recall-ranking architecture, including ItemCF and TF-IDF+FAISS recall baselines, a Two-Tower recall optimization model, and MLP/DeepFM ranking models.
- Implemented offline evaluation with Recall@K, HitRate@K, NDCG@K, and AUC to compare retrieval and ranking strategies quantitatively.
- Delivered a reproducible local demo with recommendation explanation and command-line interaction, while keeping the codebase structured for interview-ready project presentation.
