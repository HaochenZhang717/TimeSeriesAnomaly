import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_one_step_dataset(data):
    """
    data: numpy array (B, T, C)
    Returns:
        X: (B, T-1, C-1)
        Y: (B, T-1, 1)
    """
    X = data[:, :-1, :-1]
    Y = data[:, 1:, -1:]
    return X, Y


class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        y = self.proj(out)
        return y


def predictive_score_metrics(
    ori_data,
    gen_data,
    hidden_dim=64,
    max_epochs=2000,
    batch_size=64,
    patience=20,
    device="cuda"
):
    """
    Train predictor on synthetic data (with validation + early stopping),
    evaluate MAE on real data.

    Returns:
        predictive_score (float): mean MAE on original data
    """
    # ----- 1. build dataset -----
    X_gen, Y_gen = build_one_step_dataset(gen_data)
    X_real, Y_real = build_one_step_dataset(ori_data)

    X_gen = torch.tensor(X_gen, dtype=torch.float32)
    Y_gen = torch.tensor(Y_gen, dtype=torch.float32)
    X_real = torch.tensor(X_real, dtype=torch.float32)
    Y_real = torch.tensor(Y_real, dtype=torch.float32)

    train_dataset = TensorDataset(X_gen, Y_gen)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_real, Y_real)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


    # ----- 3. build model -----
    input_dim = X_gen.shape[-1]
    model = GRUPredictor(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    # ----- 4. Training with early stopping -----
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(max_epochs):

        # ------ train ------
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred = model(Xb)
            loss = loss_fn(pred, Yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ------ validation ------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = model(Xb)
                val_losses.append(loss_fn(pred, Yb).item())

        val_loss = np.mean(val_losses)

        # print progress
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.6f}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val loss = {best_val_loss:.6f}")
            break

    # ----- 5. load best model -----
    model.load_state_dict(best_model_state)
    model.eval()

    # ----- 6. evaluate on original data -----
    pred_real = []
    with torch.no_grad():
        for Xb, Yb in val_loader:
            pred_real.append(model(Xb.to(device)).cpu())
    pred_real = torch.cat(pred_real, dim=0).detach().cpu().numpy()

    # compute mean MAE
    B = X_real.shape[0]
    assert pred_real.shape == Y_real.shape
    total_mae = 0.0
    for i in range(B):
        total_mae += mean_absolute_error(Y_real[i].detach().cpu().numpy(), pred_real[i])

    predictive_score = total_mae / B
    return predictive_score