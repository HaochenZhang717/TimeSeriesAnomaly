import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def fit_classifier(
        model, normal_train_loader, anomaly_train_loader, test_loader, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_eval = float('inf')
    no_improvement = 0
    train_loss = 0
    train_seen = 0

    model.train()
    for iteration in range(100000):
        normal_inputs, normal_labels = next(normal_train_loader)
        anomaly_inputs, anomaly_labels = next(anomaly_train_loader)
        inputs = torch.cat((normal_inputs, anomaly_inputs), dim=0)
        labels = torch.cat((normal_labels, anomaly_labels), dim=0)
        loss = model.loss(inputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        train_seen += 1
        if iteration % 250 == 0 and iteration > 0:
            train_loss_avg = train_loss / train_seen
            model.eval()
            eval_loss = 0
            eval_seen = 0

            for inputs, labels in test_loader:
                with torch.no_grad():
                    loss = model.loss(inputs, labels)
                bs = labels.size(0)
                eval_loss += loss.item() * bs
                eval_seen += bs

            eval_loss = eval_loss / eval_seen
            print(f"Step{iteration} | train loss: {train_loss_avg:.4f} | eval loss: {eval_loss:.4f} ||")
            if eval_loss < best_eval:
                best_eval = eval_loss
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement > 10:
                break

            train_loss = 0
            train_seen = 0
            model.train()

    # ------------------ Collect Predictions ------------------
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        with torch.no_grad():
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
    y_prob = all_preds.detach().cpu().numpy().astype(np.float64)
    y_true = all_labels.detach().cpu().numpy().astype(np.int64)
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


    normal_mask = (y_true == 0)
    anomaly_mask = (y_true == 1)

    if normal_mask.sum() > 0:
        metrics["Normal_Accuracy"] = (y_pred[normal_mask] == 0).mean()
    else:
        metrics["Normal_Accuracy"] = float("nan")

    if anomaly_mask.sum() > 0:
        metrics["Anomaly_Accuracy"] = (y_pred[anomaly_mask] == 1).mean()
    else:
        metrics["Anomaly_Accuracy"] = float("nan")

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

    # train_set_input = torch.cat([normal, anomaly], dim=0)
    # train_set_label = torch.cat([normal_label, anomaly_label], dim=0)
    normal_train_set = TensorDataset(normal, normal_label)
    anomaly_train_set = TensorDataset(anomaly, anomaly_label)

    normal_train_loader = DataLoader(normal_train_set, batch_size=bs, shuffle=True, drop_last=True)
    anomaly_train_loader = DataLoader(anomaly_train_set, batch_size=bs, shuffle=True, drop_last=True)
    normal_train_loader = infinite_loader(normal_train_loader)
    anomaly_train_loader = infinite_loader(anomaly_train_loader)

    '''test set'''
    # test_normal = torch.tensor(test_normal_signal, dtype=torch.float32).to(device)
    # test_normal_label = torch.zeros((len(test_normal), 1), dtype=torch.float32).to(device)
    #
    # test_anomaly = torch.tensor(test_anomaly_signal, dtype=torch.float32).to(device)
    # if mode == "interval":
    #     test_anomaly_label = torch.ones((len(test_anomaly), 1), dtype=torch.float32).to(device)
    # elif mode == "timestep":
    #     test_anomaly_label = torch.tensor(test_anomaly_label, dtype=torch.float32).to(device)
    # else:
    #     raise ValueError("mode must be interval or timestep")

    test_normal = torch.tensor(normal, dtype=torch.float32).to(device)
    test_normal_label = torch.zeros((len(test_normal), 1), dtype=torch.float32).to(device)

    test_anomaly = torch.tensor(train_anomaly_signal, dtype=torch.float32).to(device)
    if mode == "interval":
        test_anomaly_label = torch.ones((len(test_anomaly), 1), dtype=torch.float32).to(device)
    elif mode == "timestep":
        test_anomaly_label = torch.tensor(test_anomaly_label, dtype=torch.float32).to(device)
    else:
        raise ValueError("mode must be interval or timestep")


    # normal_test_set = TensorDataset(test_normal, test_normal_label)
    # anomaly_test_set = TensorDataset(test_anomaly, test_anomaly_label)
    # normal_test_loader = DataLoader(normal_test_set, batch_size=bs, shuffle=False, drop_last=False)
    # anomaly_test_loader = DataLoader(anomaly_test_set, batch_size=bs, shuffle=False, drop_last=False)

    test_set_input = torch.cat([test_normal, test_anomaly], dim=0)
    test_set_label = torch.cat([test_normal_label, test_anomaly_label], dim=0)
    test_set = TensorDataset(test_set_input, test_set_label)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True)

    # print(f"test_normal.shape: {test_normal.shape}")
    # print(f"test_anomaly.shape: {test_anomaly.shape}")
    # print(f"train_normal.shape: {normal.shape}")
    # print(f"train_anomaly.shape: {anomaly.shape}")
    # breakpoint()
    metrics = fit_classifier(model, normal_train_loader, anomaly_train_loader, test_loader, lr)


    return metrics



