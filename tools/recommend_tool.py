# Re-run tool function definition after kernel reset

import pandas as pd
import numpy as np
import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# Define CTRMLP again
import torch.nn as nn

from rec_explainer import generate_explanation

class CTRMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.mlp(x)

# Load data once
df = pd.read_csv("mini_youtube8m.csv")
df["tags"] = df["tags"].apply(eval)
video_id_map = df["video_id"].tolist()

# Tag binarizer
all_tags = sorted(set(tag for tags in df["tags"] for tag in tags))
mlb = MultiLabelBinarizer(classes=all_tags)
mlb.fit(df["tags"])

# Embedding & FAISS
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
video_embeddings = np.load("video_embeddings.npy")
index = faiss.read_index("video_embeddings.faiss")

# CTR model
input_dim = len(all_tags) * 2
model = CTRMLP(input_dim)
model.load_state_dict(torch.load("ctr_mlp_model.pt", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# Helper functions
def binarize_tags(tag_list):
    vec = [0] * len(mlb.classes_)
    for tag in tag_list:
        if tag in mlb.classes_:
            vec[mlb.classes_.tolist().index(tag)] = 1
    return vec

def recommend_videos(user_tags: list[str], k: int = 5) -> list[dict]:
    print("🧪 recommend_videos 被调用！参数：", user_tags)
    if not user_tags:
        return []

    """LLM-compatible tool: Recommend videos based on user tags (keywords)"""
    user_vec = binarize_tags(user_tags)
    query_text = " ".join(user_tags)
    query_vec = model_emb.encode([query_text])
    D, I = index.search(query_vec, k * 2)

    candidate_ids = [video_id_map[idx] for idx in I[0]]
    candidate_df = df[df["video_id"].isin(candidate_ids)].reset_index(drop=True)
    video_vecs = [binarize_tags(tags) for tags in candidate_df["tags"]]
    X_pred = np.array([user_vec + v for v in video_vecs])
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

    with torch.no_grad():
        scores = model(X_pred_tensor).squeeze().numpy()

    candidate_df["score"] = scores
    sorted_df = candidate_df.sort_values(by="score", ascending=False).head(k)

    results = []

    # 推荐的视频信息 + 推荐理由
    for _, row in sorted_df.iterrows():
        explanation = generate_explanation(
            user_tags=user_tags,
            video_tags=row["tags"],
            score=row["score"],
            title=row["title"]
        )

        results.append({
            "video_id": row["video_id"],
            "title": row["title"],
            "score": round(row["score"], 4),
            "tags": row["tags"],
            "reason": explanation  # 💬 推荐理由字段
        })