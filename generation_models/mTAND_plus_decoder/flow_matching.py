from .attention_decoder import LlamaDecoder
from .time_encoder import TimeEncoder
from torch import nn
import torch



class FlowMatchingDecoder(nn.Module):
    def __init__(
            self,
            ts_dim, num_blocks, hidden_size,
            num_attention_heads, attention_dropout, intermediate_size,
            H, d_h
            ):
        super(FlowMatchingDecoder, self).__init__()
        self.decoder = LlamaDecoder(
            ts_dim, num_blocks, hidden_size, num_attention_heads,
            attention_dropout, intermediate_size
        )
        self.encoder = TimeEncoder(H, d_h, 2 * d_h)
        self.output_projection = nn.Linear(hidden_size, ts_dim)


    def forward(self, signal, ):

