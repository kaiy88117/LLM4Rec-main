import pandas as pd
import json
import random
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("mini_youtube8m.csv")
with open("user_behavior.json", "r") as f:
    user_data = json.load(f)

df["tags"] = df["tags"].apply(eval)

all_tags = set(tag for tags in df["tags"] for tag in tags)
mlb = MultiLabelBinarizer(classes=sorted(all_tags))
df_tags_encoded = pd.DataFrame(mlb.fit_transform(df["tags"]), columns=mlb.classes_)
df = pd.concat([df, df_tags_encoded], axis=1)

samples = []
for user_id, info in user_data.items():
    preferred_tags = info["preferred_tags"]
    watched = info["watched_videos"]

    for vid in watched:
        if vid in df["video_id"].values:
            row = df[df["video_id"] == vid].iloc[0]
            samples.append({"user_id": user_id, "video_id": vid, "user_tags": preferred_tags, "video_tags": row["tags"], "label": 1})

    unwatched = list(set(df["video_id"].values) - set(watched))
    neg_samples = random.sample(unwatched, k=min(len(watched)*3, len(unwatched)))
    for vid in neg_samples:
        row = df[df["video_id"] == vid].iloc[0]
        samples.append({"user_id": user_id, "video_id": vid, "user_tags": preferred_tags, "video_tags": row["tags"], "label": 0})

def binarize_tags(tag_list):
    vec = [0] * len(mlb.classes_)
    for tag in tag_list:
        if tag in mlb.classes_:
            vec[mlb.classes_.tolist().index(tag)] = 1
    return vec

output_rows = []
for s in samples:
    uvec = binarize_tags(s["user_tags"])
    vvec = binarize_tags(s["video_tags"])
    output_rows.append(uvec + vvec + [s["label"]])

columns = [f"u_{tag}" for tag in mlb.classes_] + [f"v_{tag}" for tag in mlb.classes_] + ["label"]
train_df = pd.DataFrame(output_rows, columns=columns)
train_df.to_csv("train_ctr_samples.csv", index=False)
