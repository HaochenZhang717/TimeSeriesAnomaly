import torch
from torch import nn
import math


class MultiPhi(nn.Module):
    def __init__(self, H, d_h):
        super().__init__()
        self.phis = nn.Linear(1, H * d_h)
        self.H = H
        self.d_h = d_h

    def forward(self, t):
        """
        t: (B, T)
        return: (B, T, H, d_h)
        """
        B, T = t.shape[:2]
        t = t.unsqueeze(-1) #(B, T, 1)
        out = self.phis(t) #(B, T, H * d_h)
        out = out.view(B, T, self.H, self.d_h) # (B, T, H, d_h)
        out[..., 1:] = torch.sin(out[..., 1:])

        return out


class TimeEncoder(nn.Module):
    def __init__(self, H, d_h, hidden_dim):
        super().__init__()
        self.phis_query = MultiPhi(H, d_h)
        self.phis_key = MultiPhi(H, d_h)
        self.q_projs = nn.ModuleList(
            nn.Linear(d_h, hidden_dim) for _ in range(H)
        )
        self.k_projs = nn.ModuleList(
            nn.Linear(d_h, hidden_dim) for _ in range(H)
        )


    def forward(self, query_t, key_t, key_signal, attn_mask):
        batch_size, query_len = query_t.shape
        batch_size, key_len = key_t.shape

        phi_query = self.phis_query(query_t) # (B, query_len, H, d_h)
        phi_key = self.phis_key(key_t) # (B, key_len, H, d_h)

        query = []
        for i, q_proj in enumerate(self.q_projs):
            query.append(q_proj(phi_query[:,:,i]))
        query = torch.stack(query, dim=2) # (B, query_len, H, hidden_dim)


        key = []
        for i, k_proj in enumerate(self.k_projs):
            key.append(k_proj(phi_key[:,:,i]))
        key = torch.stack(key, dim=2) # (B, key_len, H, hidden_dim)

        query = query.permute(0, 2, 1, 3) # (B, H, query_len, hidden_dim)
        key = key.permute(0, 2, 1, 3) # (B, H, key_len, hidden_dim)

        attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_len) # (B, H, query_len, key_len)
        # attn_mask # (B, key_len)
        if attn_mask is not None:
            # attn_mask: (B, key_len)
            # -> (B, 1, 1, key_len)  broadcast 到 (B, H, query_len, key_len)
            mask = attn_mask[:, None, None, :]
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1) # (B, H, query_len, key_len)
        # key_signal (B, key_len, 1)
        key_signal = key_signal.unsqueeze(1) #(B, 1, key_len, 1)
        output = attn @ key_signal #(B, H, query_len, 1)
        output = output.squeeze(-1) #(B, H, query_len)
        return output
