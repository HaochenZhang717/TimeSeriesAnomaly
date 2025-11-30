import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset, random_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.metrics import roc_auc_score




class GRUDiscriminator(nn.Module):
    """
    GRU-based binary classifier:
       input  (B, T, C)
       output (B, 1)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_state = out[:, -1, :]   # final hidden state
        logit = self.fc(last_state)
        prob = torch.sigmoid(logit)
        return prob, logit





def classification_metrics_torch(
    ori_normal_data,
    ori_anomaly_data,
    gen_anomaly_data,
    hidden_dim=64,
    max_epochs=2000,
    batch_size=64,
    patience=20,
    device="cuda"
):

    # device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 1) 准备训练集：normal + generated anomaly
    # ------------------------------
    X_train_normal = torch.tensor(ori_normal_data, dtype=torch.float32, device=device)
    X_train_fake_anom = torch.tensor(gen_anomaly_data, dtype=torch.float32, device=device)

    y_train_normal = torch.zeros((len(X_train_normal), 1), dtype=torch.float32, device=device)
    y_train_fake_anom = torch.ones((len(X_train_fake_anom), 1), dtype=torch.float32, device=device)

    X_train = torch.cat([X_train_normal, X_train_fake_anom], dim=0)
    y_train = torch.cat([y_train_normal, y_train_fake_anom], dim=0)

    train_dataset = TensorDataset(X_train, y_train)

    # ------------------------------
    # 2) 准备测试集：original normal + original anomaly
    # ------------------------------
    X_test_normal = torch.tensor(ori_normal_data, dtype=torch.float32, device=device)
    X_test_anomaly = torch.tensor(ori_anomaly_data, dtype=torch.float32, device=device)

    y_test_normal = torch.zeros((len(X_test_normal), 1), dtype=torch.float32, device=device)
    y_test_anomaly = torch.ones((len(X_test_anomaly), 1), dtype=torch.float32, device=device)

    X_test = torch.cat([X_test_normal, X_test_anomaly], dim=0)
    y_test = torch.cat([y_test_normal, y_test_anomaly], dim=0)

    test_dataset = TensorDataset(X_test, y_test)

    # ------------------------------
    # 3) DataLoader
    # ------------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ------------------------------
    # 4) 创建 classifier
    # ------------------------------
    _, T, C = X_train.shape  # (B, T, C)
    clf = GRUDiscriminator(input_dim=C, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # ------------------------------
    # 5) 训练 + early stopping
    # ------------------------------
    best_loss = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        clf.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            prob, logit = clf(xb)
            loss = criterion(prob, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(train_dataset)
        print(avg_loss)
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            best_state = clf.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break

    # load best model
    clf.load_state_dict(best_state)

    # ------------------------------
    # 6) 测试指标
    # ------------------------------
    clf.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            prob, _ = clf(xb)
            all_probs.append(prob.cpu())
            all_labels.append(yb)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 分类正确情况
    preds = (all_probs > 0.5).astype(int)

    accuracy = (preds == all_labels).mean()

    # 分别计算 normal 和 anomaly 的测试集 accuracy
    normal_mask = (all_labels == 0).squeeze()
    anomaly_mask = (all_labels == 1).squeeze()

    normal_acc = (preds[normal_mask] == 0).mean()
    anomaly_acc = (preds[anomaly_mask] == 1).mean()

    # AUC（可选）
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = None

    return {
        "test_accuracy": float(accuracy),
        "test_normal_acc": float(normal_acc),
        "test_anomaly_acc": float(anomaly_acc),
        "auc": None if auc is None else float(auc)
    }