import torch
from torch import nn
from typing import Optional
import math
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,

):
    # key_states = repeat_kv(key, module.num_key_value_groups)
    # value_states = repeat_kv(value, module.num_key_value_groups)
    key_states = key
    value_states = value

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout):
        super().__init__()
        # self.config = config
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None:
            # attention_mask: (B, 1, D, D)
            device = hidden_states.device
            dtype = hidden_states.dtype
            B, D = hidden_states.shape[:2]
            E = encoder_states.shape[1]

            # (1) encoder -> encoder : allow
            enc_enc = torch.zeros(
                (B, 1, E, E), device=device, dtype=dtype
            )

            # (2) encoder -> decoder : block
            enc_dec = torch.full(
                (B, 1, E, D), float("-inf"), device=device, dtype=dtype
            )

            # (3) decoder -> encoder : allow
            dec_enc = torch.zeros(
                (B, 1, D, E), device=device, dtype=dtype
            )

            # (4) decoder -> decoder : use provided mask
            dec_dec = attention_mask  # (B, 1, D, D)

            # 拼起来
            upper = torch.cat([enc_enc, enc_dec], dim=-1)
            lower = torch.cat([dec_enc, dec_dec], dim=-1)
            attention_mask = torch.cat([upper, lower], dim=-2)

        output_length = hidden_states.shape[1]

        hidden_states = torch.cat([encoder_states, hidden_states], dim=1)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) #(B, num_heads, len, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings  # expected shape: (B, D, head_dim)

            # after concat: seq_len = E + D
            # query_states/key_states shape: (B, nheads, E+D, head_dim)
            # only rotate the decoder slice [E:].
            E = encoder_states.shape[1]
            q_dec = query_states[:, :, E:, :]
            k_dec = key_states[:, :, E:, :]

            q_dec, k_dec = apply_rotary_pos_emb(q_dec, k_dec, cos, sin)

            query_states = torch.cat([query_states[:, :, :E, :], q_dec], dim=2)
            key_states = torch.cat([key_states[:, :, :E, :], k_dec], dim=2)

        # cos, sin = position_embeddings
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = attn_output[:, -output_length:, :]

        return attn_output, attn_weights


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = LlamaAttention(hidden_size, num_attention_heads, attention_dropout)

        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.encoder_norm = LlamaRMSNorm(hidden_size)

        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=1e-6)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(timestep).chunk(6, dim=1)

        residual = hidden_states

        hidden_states = modulate(self.input_layernorm(hidden_states), shift_msa, scale_msa)
        encoder_states = self.encoder_norm(encoder_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            encoder_states=encoder_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = gate_msa.unsqueeze(1) * hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = modulate(self.post_attention_layernorm(hidden_states), shift_mlp, scale_mlp)
        hidden_states = self.mlp(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaDecoder(nn.Module):
    def __init__(self, ts_dim, num_blocks, hidden_size, num_attention_heads, attention_dropout, intermediate_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(hidden_size, num_attention_heads, attention_dropout, intermediate_size) for _ in range(num_blocks)]
        )

        self.output_head = nn.Linear(hidden_size, ts_dim)

        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=8192,
        )

        config.rope_parameters = {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.norm = LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.t_embedder = TimestepEmbedder(hidden_size)

    def forward(
            self,
            timestep: torch.Tensor,
            hidden_states: torch.Tensor,
            encoder_states: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        timestep = self.t_embedder(timestep)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                timestep=timestep,
                hidden_states=hidden_states,
                encoder_states=encoder_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings
            )
        hidden_states = self.norm(hidden_states)

        return hidden_states

