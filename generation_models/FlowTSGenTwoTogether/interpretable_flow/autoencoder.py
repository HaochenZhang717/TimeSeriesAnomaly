from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise ValueError(f"activation should be 'relu' or 'gelu', not '{activation}'")

# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding
    elif pos_encoding == "continuous":
        return ContinuousPositionalEncoding
    else:
        raise NotImplementedError(f"pos_encoding should be 'learnable', 'fixed', or 'continuous', not '{pos_encoding}'")


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, d_model)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        masked_X = X.clone()
        masked_X[torch.isnan(masked_X)] = 0

        inp = masked_X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)

        return output


class ContinuousPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ContinuousPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x, times):
        """
        Args:
            x: Tensor of shape (seq_length, batch_size, d_model)
            times: Tensor of shape (seq_length, batch_size)
        """
        times = times.unsqueeze(-1)  # (seq_length, batch_size, 1)
        # Create the position encodings using times
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
        pe = times * div_term  # (seq_length, batch_size, d_model // 2)
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # (seq_length, batch_size, d_model)
        x = x + pe
        return self.dropout(x)


class TST_Decoder(nn.Module):
    def __init__(self, inp_dim, hidden_dim, layers, ts_channels):  # Fixed the method name
        super(TST_Decoder, self).__init__()  # Fixed the method name
        self.z_dim = inp_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(
            input_size=self.z_dim,
            hidden_size=self.hidden_dim,
            bidirectional=True,
            num_layers=layers,
            batch_first=True
        )

        self.linear = nn.Linear(self.hidden_dim * 2, ts_channels)

    def forward(self, z):
        # Decode
        h, _ = self.rnn(z)
        x_hat = F.sigmoid(self.linear(h))
        return x_hat



class AutoEncoder(nn.Module):
    def __init__(
            self,
            feat_dim, max_len, d_model, n_heads,
            num_layers, dim_feedforward, dropout,
            pos_encoding, activation,
            norm, freeze,
            decoder_inp_dim,
            decoder_hidden_dim,
            decoder_layers
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = TSTransformerEncoder(
            feat_dim, max_len, d_model, n_heads,
            num_layers, dim_feedforward, dropout,
            pos_encoding, activation,
            norm, freeze,
        )

        self.decoder = TST_Decoder(
            decoder_inp_dim,
            decoder_hidden_dim,
            decoder_layers,
            feat_dim
        )

    def forward(self, x, anomaly_label):
        inverse_anomaly_label = 1 - anomaly_label
        x = x * inverse_anomaly_label.unsqueeze(-1)

        h = self.encoder(x, padding_masks=(anomaly_label==0))
        x_tilde = self.decoder(h)

        loss = F.mse_loss(
            x_tilde[anomaly_label == 0],
            x[anomaly_label == 0]
        )
        return x_tilde, loss


def fast_build_autoencoder(feat_dim, max_len):
    hidden_dim=40
    return AutoEncoder(
        feat_dim,
        max_len,
        d_model=hidden_dim, n_heads=5,
        num_layers=6, dim_feedforward=2048,
        dropout=0.1,
        pos_encoding='fixed',
        activation='gelu',
        norm='BatchNorm',
        freeze=False,
        decoder_inp_dim=40,
        decoder_hidden_dim=int(hidden_dim + (feat_dim - hidden_dim) / 2),
        decoder_layers=3
    )


#
#
# tst_config = {
#             'feat_dim': args.input_size,
#             'max_len': args.seq_len,
#             'd_model': args.hidden_dim,
#             'n_heads': args.n_heads,  # Number of attention heads
#             'num_layers': args.num_layers,  # Number of transformer layers
#             'dim_feedforward': args.dim_feedforward,
#             'dropout': args.dropout,
#             'pos_encoding': args.pos_encoding,  # or 'learnable'
#             'activation': args.activation,
#             'norm': args.norm,
#             'freeze': args.freeze
#         }
#
#
# from omegaconf import OmegaConf
# import argparse
#
# def parse_args_irregular():
#     """
#     Parse arguments for unconditional models
#     Returns: unconditioanl generation args namespace
#
#     """
#     parser = argparse.ArgumentParser()
#     # --- general ---
#     # NOTE: the following arguments are general, they are not present in the config file:
#     parser.add_argument('--seed', type=int, default=0, help='random seed')
#     parser.add_argument('--num_workers', default=4, type=int,
#                         help='Number of workers to use for dataloader')
#     parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
#     parser.add_argument('--log_dir', default='./logs', help='path to save logs')
#     parser.add_argument('--neptune', type=bool, default=False, help='use neptune logger')
#     parser.add_argument('--missing_rate', type=float, default=0.3)
#     parser.add_argument('--tags', type=str, default=['30 missing rate'], help='tags for neptune logger', nargs='+')
#
#     # --- diffusion process --- #
#     parser.add_argument('--beta1', type=float, default=1e-5, help='value of beta 1')
#     parser.add_argument('--betaT', type=float, default=1e-2, help='value of beta T')
#     parser.add_argument('--deterministic', action='store_true', default=False, help='deterministic sampling')
#
#     # ## --- config file --- # ##
#     # NOTE: the below configuration are arguments. if given as CLI argument, they will override the config file values
#     parser.add_argument('--config', type=str, default='./configs/seq_len_24/stock.yaml', help='config file')
#     parser.add_argument('--model_save_path', type=str, default='./saved_models', help='path to save the model')
#
#
#     # --- training ---
#     parser.add_argument('--epochs', type=int, help='number of epochs to train')
#     parser.add_argument('--batch_size', type=int, help='training batch size')
#     parser.add_argument('--learning_rate', type=float, help='learning rate')
#     parser.add_argument('--weight_decay', type=float, help='weight decay')
#
#     # --- data ---:
#     parser.add_argument('--dataset',
#                         choices=['sine', 'energy', 'mujoco', 'stock', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity'], help='training dataset')
#
#     parser.add_argument('--seq_len', type=int,
#                         help='input sequence length,'
#                              ' only needed if using short-term datasets(stocks,sine,energy,mujoco)')
#
#     # --- image transformations ---:
#     parser.add_argument('--delay', type=int,
#                         help='delay for the delay embedding transformation, only needed if using delay embedding')
#     parser.add_argument('--embedding', type=int,
#                         help='embedding for the delay embedding transformation, only needed if using delay embedding')
#
#     # --- model--- :
#     parser.add_argument('--img_resolution', type=int, help='image resolution')
#     parser.add_argument('--input_channels', type=int,
#                         help='number of image channels, 2 if stft is used, 1 for delay embedding')
#     parser.add_argument('--unet_channels', type=int, help='number of unet channels')
#     parser.add_argument('--ch_mult', type=int, help='ch mut', nargs='+')
#     parser.add_argument('--attn_resolution', type=int, help='attn_resolution', nargs='+')
#     parser.add_argument('--diffusion_steps', type=int, help='number of diffusion steps')
#     parser.add_argument('--ema', type=bool, help='use ema')
#     parser.add_argument('--ema_warmup', type=int, help='ema warmup')
#
#     # --- TST ---
#     parser.add_argument('--hidden_dim', type=int, default=40, help='dimension of the hidden layer')
#     parser.add_argument('--r_layer', type=int, default=2, help='number of RNN layers')
#     parser.add_argument('--last_activation_r', type=str, default='sigmoid', help='last activation function for RNN layers')
#     parser.add_argument('--first_epoch', type=int, default=2, help='number of first epoch to start training')
#     parser.add_argument('--x_hidden', type=int, default=48, help='dimension of x hidden layer')
#     parser.add_argument('--input_size', type=int, default=1, help='input size of the model')
#
#     # Adding new arguments for tst_config
#     parser.add_argument('--n_heads', type=int, default=5, help='number of attention heads')
#     parser.add_argument('--num_layers', type=int, default=6, help='number of transformer layers')
#     parser.add_argument('--dim_feedforward', type=int, default=2048, help='dimension of feedforward layers')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
#     parser.add_argument('--pos_encoding', type=str, choices=['fixed', 'learnable'], default='fixed',
#                         help='positional encoding type')
#     parser.add_argument('--activation', type=str, choices=['relu', 'gelu'], default='gelu', help='activation function')
#     parser.add_argument('--norm', type=str, choices=['BatchNorm', 'LayerNorm'], default='BatchNorm',
#                         help='normalization type')
#     parser.add_argument('--freeze', type=bool, default=False, help='freeze transformer layers')
#
#     parser.add_argument('--ts_rate', type=float, default=0, help='teacher forcing rate for tst')
#     parser.add_argument('--save_model', type=bool, default=False, help='save model')
#     parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='noise level injected to the original data')
#     parser.add_argument('--new_metrics', type=int, default=1, help='save model')
#
#     # --- logging ---s
#     parser.add_argument('--logging_iter', type=int, default=10, help='number of iterations between logging')
#     parser.add_argument('--percent', type=int, default=100)
#     parsed_args = parser.parse_args()
#
#     # load config file
#     config = OmegaConf.to_object(OmegaConf.load(parsed_args.config))
#     # override config file with command line args
#     for k, v in vars(parsed_args).items():
#         if v is None:
#             setattr(parsed_args, k, config.get(k, None))
#     # add to the parsed args, configs that are not in the parsed args but do in the config file
#     # this is needed since multiple config files setups may be used
#     for k, v in config.items():
#         if k not in vars(parsed_args):
#             setattr(parsed_args, k, v)
#
#     parsed_args.input_size = parsed_args.input_channels
#     return parsed_args