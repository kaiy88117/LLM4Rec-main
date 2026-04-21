# Interview Notes

## 1-Minute Version

I built a MovieLens 1M recommendation project with a modular recall-ranking architecture. On the recall side, I implemented ItemCF and TF-IDF plus FAISS as baselines, then added a Two-Tower model as the recall optimization route. On the ranking side, I used a PyTorch MLP baseline and a DeepFM optimization version. I also built offline evaluation with Recall@K, HitRate@K, NDCG@K, and AUC, and used the results to compare methods quantitatively. The project also includes explanation generation and a command-line demo, so it is both experimentally grounded and locally runnable.

## 3-Minute Version

I treated the project as a standard recommendation-system pipeline and split it into recall, ranking, explanation, and online entry.

First, I migrated the original demo structure to MovieLens 1M and built a stable local pipeline. That included data preprocessing, recall artifacts, ranking samples, model training, and a command-line chat entry.

Second, I made the recall layer comparable instead of relying on a single route. I added:

- ItemCF as the collaborative filtering baseline
- TF-IDF plus FAISS as the content recall baseline
- Two-Tower as the recall optimization version

Third, I kept the ranking layer simple and controlled. I used:

- MLP as the ranking baseline
- DeepFM as the ranking optimization version

Fourth, I implemented offline evaluation with Recall@K, HitRate@K, NDCG@K, and AUC. That allowed me to compare recall and ranking choices quantitatively rather than only showing demo outputs.

The current results show that ItemCF is still the strongest recall route on MovieLens 1M, while DeepFM improves over MLP on ranking. Two-Tower has been integrated successfully as a trainable recall route, but under the current lightweight feature design it does not beat ItemCF yet. I think that is still a useful result, because it shows the project is evaluation-driven instead of only stacking models.

## Likely Questions

### 1. Why did ItemCF beat Two-Tower?

Answer direction:

- MovieLens is heavily behavior-driven, so collaborative signals are strong
- current Two-Tower features are still simple, mostly genre-level signals
- trainable recall models need stronger features or better sampling to fully show gains
- this result is realistic and still valuable because it shows proper baseline comparison

### 2. Why keep TF-IDF + FAISS if it is weaker?

Answer direction:

- it serves as a clean content baseline
- it is interpretable and easy to reproduce
- it helps explain the difference between collaborative recall and content recall

### 3. Why use MLP and DeepFM instead of more complex ranking models?

Answer direction:

- I wanted a controlled project scope
- MLP is a clean baseline
- DeepFM is a standard next-step upgrade for feature interaction modeling
- adding too many ranking families would make the comparison less focused

### 4. What does AUC mean in your project?

Answer direction:

- for ranking evaluation, I sample negatives and compare model scores on held-out positives versus negatives
- AUC measures how well the model separates positive from negative candidates
- I use it together with top-K ranking metrics because they reflect different aspects of model quality

### 5. How would you improve the project next?

Answer direction:

- strengthen Two-Tower features with IDs, histories, and metadata
- improve content recall with stronger encoders
- add better experiment tracking and more robust evaluation splits
- enrich ranking features with sequential behavior

### 6. What is the main engineering value of this project?

Answer direction:

- the project is modular
- baseline and optimized versions are clearly separated
- the system can be reproduced locally
- online demo and offline evaluation are both present

## What To Emphasize

- recommendation systems should be discussed as recall plus ranking, not as one model
- baseline quality matters; optimization must be justified by metrics
- the project demonstrates both system design and experimental reasoning
- negative results are still useful when they are backed by fair comparison

## Short Resume Pitch

I built a MovieLens 1M recommendation project with three recall routes, two ranking models, offline evaluation, and a local demo. The project compares ItemCF, TF-IDF plus FAISS, and Two-Tower for recall, and compares MLP versus DeepFM for ranking, using Recall@K, HitRate@K, NDCG@K, and AUC.
