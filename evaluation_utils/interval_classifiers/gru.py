import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    GRU-based binary classifier:
       input  (B, T, C)
       output (B, 1)
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        # print("out.shape", out.shape)
        # print("x.shape", x.shape)
        last_state = out[:, -1, :]   # final hidden state
        logit = self.fc(last_state)
        prob = torch.sigmoid(logit)
        return prob, logit

    def loss(self, x, labels):
        criterion = torch.nn.BCELoss()
        prob, logit = self.forward(x)
        loss = criterion(prob, labels)
        return loss

    def run_inference(self, labels):
        prob, logit = self.forward(labels)
        return prob
