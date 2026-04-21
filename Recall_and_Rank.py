from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import json
import torch
from train_ctr import CTRMLP  # 假设模型结构保存在 train_ctr.py 中
from sklearn.preprocessing import MultiLabelBinarizer

# ================== Part 1: 加载视频数据和嵌入 ==================
df = pd.read_csv("mini_youtube8m.csv")
df["tags"] = df["tags"].apply(eval)
df["full_text"] = df["title"] + ". " + df["description"] + ". " + df["tags"]

video_id_map = df["video_id"].tolist()
video_embeddings = np.load("video_embeddings.npy")
index = faiss.read_index("video_embeddings.faiss")

model_emb = SentenceTransformer('all-MiniLM-L6-v2')

# ================== Part 2: 加载用户数据 ==================
with open("user_behavior.json", "r") as f:
    user_data = json.load(f)

user = user_data["user_1"]
user_query = " ".join(user["preferred_tags"])

# ================== Part 3: FAISS 检索召回 Top-K ==================
query_vec = model_emb.encode([user_query])
D, I = index.search(query_vec, 10)
candidate_ids = [video_id_map[idx] for idx in I[0]]
candidate_df = df[df["video_id"].isin(candidate_ids)].reset_index(drop=True)

# ================== Part 4: MLP 排序器 ==================

# 标签 binarizer（与训练阶段保持一致）
all_tags = sorted(set(tag for tags in df["tags"] for tag in tags))
mlb = MultiLabelBinarizer(classes=all_tags)
mlb.fit(df["tags"])

def binarize_tags(tag_list):
    vec = [0] * len(mlb.classes_)
    for tag in tag_list:
        if tag in mlb.classes_:
            vec[mlb.classes_.tolist().index(tag)] = 1
    return vec

user_vec = binarize_tags(user["preferred_tags"])
video_vecs = [binarize_tags(tags) for tags in candidate_df["tags"]]
X_pred = np.array([user_vec + v for v in video_vecs])
X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32)

# 加载训练好的模型（强制使用 CPU）
input_dim = X_pred.shape[1]
model = CTRMLP(input_dim)
model.load_state_dict(torch.load("ctr_mlp_model.pt", map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    scores = model(X_pred_tensor).squeeze().numpy()

candidate_df["score"] = scores
sorted_df = candidate_df.sort_values(by="score", ascending=False)

# ================== Part 5: 输出推荐结果 ==================
print("🎯 用户兴趣关键词:", user_query)
print("🔽 推荐视频（按点击率预测排序）:")
for _, row in sorted_df.iterrows():
    print(f"- {row['video_id']}: {row['title']} | tags={row['tags']} | score={row['score']:.4f}")
