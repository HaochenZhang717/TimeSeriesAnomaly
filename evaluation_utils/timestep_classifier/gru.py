import torch
import torch.nn as nn

class GRUTimestepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        logits = self.fc(out)   # (B, T, 1)
        probs = torch.sigmoid(logits)
        return probs, logits

    def loss(self, x, labels):
        _, logits = self.forward(x)
        # BCEWithLogitsLoss = sigmoid + BCE in one op (more stable)
        criterion = nn.BCEWithLogitsLoss()
        # reshape logits: (B, T)
        logits = logits.squeeze(-1)
        # reshape label: (B, T)
        labels = labels.float().squeeze(-1)
        return criterion(logits, labels)

    def run_inference(self, labels):
        probs, logits = self.forward(labels)
        return probs
