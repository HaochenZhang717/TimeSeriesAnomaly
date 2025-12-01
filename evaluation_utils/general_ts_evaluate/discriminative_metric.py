import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split


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


def discriminative_score_metrics(
    ori_data,
    gen_data,
    hidden_dim=64,
    max_epochs=2000,
    batch_size=64,
    patience=20,
    device="cuda"
):
    """
    PyTorch version of discriminative score in TimeGAN,
    rewritten to match the structure/style of predictive_score_metrics_torch.

    Train discriminator on synthetic+real data (with validation + early stopping),
    evaluate accuracy on a held-out test set.

    Returns:
        disc_score = |accuracy - 0.5|
        real_acc
        fake_acc
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ----------- 1. build full dataset -----------
    # label: real=1, fake=0
    X_real = torch.tensor(ori_data, dtype=torch.float32)
    X_fake = torch.tensor(gen_data, dtype=torch.float32)

    y_real = torch.ones((X_real.shape[0], 1), dtype=torch.float32)
    y_fake = torch.zeros((X_fake.shape[0], 1), dtype=torch.float32)

    X_all = torch.cat([X_real, X_fake], dim=0)
    y_all = torch.cat([y_real, y_fake], dim=0)

    # shuffle dataset
    n = X_all.shape[0]
    idx = torch.randperm(n)
    X_all = X_all[idx]
    y_all = y_all[idx]

    # ----------- 2. train/val/test split -----------
    n_test = int(0.2 * n)
    n_val  = int(0.2 * n)
    n_train = n - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        TensorDataset(X_all, y_all),
        [n_train, n_val, n_test]
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ----------- 3. build model -----------
    input_dim = ori_data.shape[-1]
    model = GRUDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # ----------- 4. training with early stopping -----------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):

        # ---- train ----
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            _, logit = model(Xb)
            loss = loss_fn(logit, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- validation ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                _, logit = model(Xb)
                val_losses.append(loss_fn(logit, yb).item())

        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.6f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val_loss = {best_val_loss:.6f}")
            break

    # ----------- 5. load best model -----------
    model.load_state_dict(best_state)
    model.eval()

    # ----------- 6. evaluate on test set -----------
    y_true = []
    y_pred = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            prob, _ = model(Xb.to(device))
            pred = (prob.cpu().numpy().flatten() > 0.5).astype(int)
            y_pred.extend(pred.tolist())
            y_true.extend(yb.cpu().numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- compute accuracies ----
    overall_acc = accuracy_score(y_true, y_pred)

    real_mask = (y_true == 1)
    fake_mask = (y_true == 0)

    real_acc = accuracy_score(y_true[real_mask], y_pred[real_mask])
    fake_acc = accuracy_score(y_true[fake_mask], y_pred[fake_mask])

    disc_score = abs(overall_acc - 0.5)

    return disc_score, fake_acc, real_acc