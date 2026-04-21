import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train_ctr_samples.csv")
X = train_df.drop(columns=["label"]).values
y = train_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32)

class CTRModel(nn.Module):
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

model = CTRModel(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).numpy()
        val_labels = y_val_tensor.numpy()
        auc = roc_auc_score(val_labels, val_preds)
        logloss = log_loss(val_labels, val_preds)
        print(f"Epoch {epoch+1}: AUC = {auc:.4f}, Log Loss = {logloss:.4f}")

# ✅ 保存模型
torch.save(model.state_dict(), "ctr_mlp_model.pt")
print("✅ 模型已保存为 ctr_mlp_model.pt")