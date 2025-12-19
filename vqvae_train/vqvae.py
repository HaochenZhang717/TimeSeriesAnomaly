import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE (nearest neighbor) with straight-through gradient.

    z_e: [B, T', D]  (encoder outputs)
    z_q: [B, T', D]  (quantized)
    ids: [B, T']     (code indices)
    loss: vq loss (codebook + commitment)
    """
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        B, T, D = z_e.shape
        assert D == self.code_dim

        flat = z_e.reshape(B * T, D)  # [BT, D]
        e = self.codebook.weight      # [K, D]

        # squared euclidean distances: ||x||^2 - 2 x·e + ||e||^2
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ e.t()
            + e.pow(2).sum(dim=1, keepdim=True).t()
        )  # [BT, K]

        ids = torch.argmin(dist, dim=1)             # [BT]
        z_q = self.codebook(ids).view(B, T, D)      # [B, T, D]

        # VQ losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss   = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, ids.view(B, T), vq_loss


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        ts_dim: int,
        code_dim: int,
        config: LlamaConfig,
        num_class_tokens: int,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_class_tokens = num_class_tokens

        # ---- input projection (time series → hidden) ----
        self.input_proj = nn.Conv1d(
            ts_dim, config.hidden_size, kernel_size=3, padding=1
        )

        # ---- class tokens ----
        self.class_tokens = nn.Parameter(
            torch.randn(num_class_tokens, config.hidden_size)
        )

        # ---- transformer blocks ----
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        # ---- rotary embedding ----
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # ---- final norm ----
        self.norm = LlamaRMSNorm(code_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(config.hidden_size, code_dim)

    def forward(self, x, attention_mask):
        """
        x: (B, T, ts_dim)
        """
        B, T, _ = x.shape
        device = x.device

        # ---- project input ----
        x = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
        # (B, T, H)

        # ---- prepend class tokens ----
        cls = self.class_tokens.unsqueeze(0).expand(B, -1, -1)
        hidden_states = torch.cat([cls, x], dim=1)
        # (B, N_cls + T, H)

        seq_len = hidden_states.shape[1]

        # ---- attention mask  ----
        extra_attention_mask = torch.ones(B, 1, 1, self.num_class_tokens, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([extra_attention_mask, attention_mask], dim=-1)

        # ---- position ids (class tokens share pos=0) ----
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)


        # ---- transformer blocks ----
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                output_attentions=False,
            )[0]


        # ---- split outputs ----
        cls_out = hidden_states[:, : self.num_class_tokens]
        cls_out = self.out_proj(cls_out)
        cls_out = self.norm(cls_out)

        return cls_out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        code_dim: int,
        config: LlamaConfig,
        ts_dim: int,
        max_len: int
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.max_len = max_len

        # time query tokens (NOT class tokens)
        self.time_queries = nn.Parameter(
            torch.randn(max_len, config.hidden_size)
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.output_proj = nn.Linear(config.hidden_size, ts_dim)
        self.input_proj = nn.Linear(code_dim, config.hidden_size)


    def forward(self, z, attention_mask):
        """
        z: (B, K, H)   latent tokens (K=4)
        T: int         output length (e.g. 1000)
        attention_mask: (B, 1, 1, T)
        """
        T = attention_mask.shape[-1]
        B, num_latent_tokens = z.shape[:2]
        device = z.device
        z = self.input_proj(z) # (B, K, hidden_dim)

        hidden_states = self.time_queries[:T].unsqueeze(0).expand(B, -1, -1)# (B, T, hidden_dim)
        hidden_states = torch.cat([z, hidden_states], dim=1) # (B, K+T, hidden_dim)

        input_length = hidden_states.shape[1]
        # ---- rotary embeddings ----
        position_ids = torch.arange(input_length, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)


        # ---- attention mask ----
        extra_attention_mask = torch.ones(B, 1, 1, num_latent_tokens, device=device)
        attention_mask = torch.cat([extra_attention_mask, attention_mask], dim=-1) #(B, 1, 1, T+num_latent_tokens)
        # ---- decoder blocks ----
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                encoder_hidden_states=z,   # ← latent conditioning
                use_cache=False,
            )[0]

        hidden_states = self.norm(hidden_states)

        # ---- project back to signal ----
        out = self.output_proj(hidden_states)  # (B, T, ts_dim)
        return out[:, num_latent_tokens:]


class VQVAE(nn.Module):
    """
    Standard VQ-VAE (nearest neighbor) with straight-through gradient.

    z_e: [B, T', D]  (encoder outputs)
    z_q: [B, T', D]  (quantized)
    ids: [B, T']     (code indices)
    loss: vq loss (codebook + commitment)
    """
    def __init__(
            self,
            ts_dim: int,
            num_class_tokens: int,
            encoder_config: LlamaConfig,

            num_codes: int,
            code_dim: int,
            beta: float,

            decoder_config: LlamaConfig,
            max_len: int
    ):
        super().__init__()


        self.quantizer = VectorQuantizer(
            num_codes, code_dim, beta
        )

        self.encoder = TransformerEncoder(
            ts_dim,
            code_dim,
            encoder_config,
            num_class_tokens
        )

        self.decoder = TransformerDecoder(
            code_dim,
            decoder_config,
            ts_dim,
            max_len
        )

    def forward(self, x, attention_mask):
        z_e = self.encoder(x, attention_mask)
        z_q, ids, vq_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q, attention_mask)
        return x_hat, ids, vq_loss


def test_encoder():
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )

    encoder = TransformerEncoder(
        ts_dim=3,
        code_dim=8,
        config=config,
        num_class_tokens=4,
    )

    x = torch.randn(2, 128, 3)  # (B, T, ts_dim)
    attention_mask = torch.ones(2, 1, 1, 128)

    out = encoder(x, attention_mask)

    print("cls_tokens:", out.shape)
    # print("seq_tokens:", out["seq_tokens"].shape)



def test_decoder():

    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )


    decoder = TransformerDecoder(
        code_dim=8,
        ts_dim=3,
        config=config,
        max_len=100,
    )

    # 假装这是 VQ 之后的 latent tokens
    seq_tokens = torch.randn(2, 4, 8)
    attn_mask = torch.ones(2,1,1,100)
    out = decoder(seq_tokens, attn_mask)

    print("recon:", out.shape)  # (2, 128, 3)


def test_vqvae():
    encoder_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )

    decoder_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )

    vqvae = VQVAE(
            ts_dim=1, num_class_tokens=4,
            encoder_config=encoder_config,
            num_codes=200, code_dim=8, beta=0.25,
            decoder_config=decoder_config,
            max_len=1000)
    input_series = torch.randn(2, 100, 1)
    attention_mask = torch.ones(2, 1, 1, 100)
    out = vqvae(input_series, attention_mask)



def build_vqvae(
        ts_dim, num_class_tokens, code_dim,
        codebook_size, max_len, beta
):
    encoder_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )

    decoder_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )

    vqvae = VQVAE(
            ts_dim=ts_dim, num_class_tokens=num_class_tokens,
            encoder_config=encoder_config,
            num_codes=codebook_size, code_dim=code_dim, beta=beta,
            decoder_config=decoder_config,
            max_len=max_len)
    return vqvae

if __name__ == "__main__":
    # test_encoder()
    # test_decoder()
    test_vqvae()

