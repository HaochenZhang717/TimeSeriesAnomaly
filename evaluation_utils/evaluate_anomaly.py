import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

def fit_classifier(model, train_loader, test_loader, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_eval = float('inf')
    no_improvement = 0
    for epoch in tqdm(range(1000), desc="Training Classifier"):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            loss = model.loss(inputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        eval_loss = 0
        eval_seen = 0

        for inputs, labels in test_loader:
            loss = model.loss(inputs)
            bs = labels.size(0)
            eval_loss += loss.item() * bs
            eval_seen += bs

        eval_loss = eval_loss / eval_seen
        if eval_loss < best_eval:
            best_eval = eval_loss
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement > 10:
            break

    # ------------------ Collect Predictions ------------------
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        preds = model.run_inference(inputs)  # probabilities
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    # (num_batches, B, ... ) → concat → (N, ...)
    all_preds = torch.cat(all_preds, dim=0).squeeze()
    all_labels = torch.cat(all_labels, dim=0).squeeze()

    # If time-step: flatten (B,T) → (B*T,)
    if all_preds.dim() == 2:
        all_preds = all_preds.reshape(-1)
        all_labels = all_labels.reshape(-1)

    # Convert to numpy
    y_prob = all_preds.numpy().astype(np.float64)
    y_true = all_labels.numpy().astype(np.int64)
    y_pred = (y_prob >= 0.5).astype(int)

    # ------------------ Metrics ------------------
    metrics = {}

    # AU-ROC: only valid if both classes exist
    if len(np.unique(y_true)) == 2:
        metrics["AUROC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["AUROC"] = float("nan")

    # AU-PR
    metrics["AUPR"] = average_precision_score(y_true, y_prob)

    # F1
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)

    return metrics



def run_anomaly_quality_test(
        train_normal_signal, train_anomaly_signal, train_anomaly_label,
        test_normal_signal, test_anomaly_signal, test_anomaly_label,
        model, device, lr, bs, mode
):
    '''original train set'''
    normal = torch.tensor(train_normal_signal, dtype=torch.float32).to(device)
    normal_label = torch.zeros((len(normal), 1), dtype=torch.float32).to(device)

    anomaly = torch.tensor(train_anomaly_signal, dtype=torch.float32).to(device)
    if mode == "interval":
        anomaly_label = torch.ones((len(anomaly), 1), dtype=torch.float32).to(device)
    elif mode == "timestep":
        anomaly_label = torch.tensor(train_anomaly_label, dtype=torch.float32).to(device)
    else:
        raise ValueError("mode must be interval or timestep")

    train_set_input = torch.cat([normal, anomaly], dim=0)
    train_set_label = torch.cat([normal_label, anomaly_label], dim=0)
    train_set = TensorDataset(train_set_input, train_set_label)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)

    '''test set'''
    test_normal = torch.tensor(test_normal_signal, dtype=torch.float32).to(device)
    test_normal_label = torch.zeros((len(test_normal), 1), dtype=torch.float32).to(device)

    test_anomaly = torch.tensor(test_anomaly_signal, dtype=torch.float32).to(device)
    if mode == "interval":
        test_anomaly_label = torch.ones((len(test_anomaly), 1), dtype=torch.float32).to(device)
    elif mode == "timestep":
        test_anomaly_label = torch.tensor(test_anomaly_label, dtype=torch.float32).to(device)
    else:
        raise ValueError("mode must be interval or timestep")



    test_set_input = torch.cat([test_normal, test_anomaly], dim=0)
    test_set_label = torch.cat([test_normal_label, test_anomaly_label], dim=0)

    test_set = TensorDataset(test_set_input, test_set_label)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True)
    # print(f"test_normal.shape: {test_normal.shape}")
    # print(f"test_anomaly.shape: {test_anomaly.shape}")
    # print(f"train_normal.shape: {normal.shape}")
    # print(f"train_anomaly.shape: {anomaly.shape}")
    # breakpoint()
    metrics = fit_classifier(model, train_loader, test_loader, lr)

    return metrics


