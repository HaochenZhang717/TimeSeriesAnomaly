import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from .transformer import Transformer
import os
import random


class FM_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)


        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))

    def prepare_for_finetune(self, ckpt_path):
        # data = {
        #     'step': self.step,
        #     'model': self.model.state_dict(),
        #     'ema': self.ema.state_dict(),
        #     'opt': self.opt.state_dict(),
        # }
        # load checkpoint first
        # ckpt_dict = ckpt_dict['ema']
        # new_ckpt_dict = {}
        # for k, v in ckpt_dict.items():
        #     new_k = k.replace('ema_model.', '')  # 去掉 prefix
        #     new_ckpt_dict[new_k] = v

        if ckpt_path != "none":
            model_device = next(self.parameters()).device
            ckpt_dict = torch.load(ckpt_path, map_location=model_device, weights_only=False)
            self.load_state_dict(ckpt_dict)
            print("pretrained checkpoint loaded")
        self.model.prepare_for_finetune()

    def output(self, x, t, anomaly_label, padding_masks=None):

        output = self.model(x, t, anomaly_label, padding_masks=None)

        return output


    @torch.no_grad()
    def sample(self, shape, anomaly_label):
        self.eval()
        zt = torch.randn(shape).cuda()  ## init the noise
        ## t shifting from stable diffusion 3
        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)
        random_index = random.randint(0, anomaly_label.shape[0]-1)
        model_device = next(self.parameters()).device
        anomaly_label = anomaly_label[random_index].to(model_device, dtype=torch.long)
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(
                zt.clone(),
                torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).cuda().squeeze(),
                anomaly_label=anomaly_label,
                padding_masks=None
            )
            zt = zt.clone() + step * v 

        return zt 


    def generate_mts(self, anomaly_label, batch_size=16, model_kwargs=None, cond_fn=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        return self.sample((batch_size, seq_length, feature_size), anomaly_label)


    def _train_loss(self, x_start, anomaly_label):
        
        z0 = torch.randn_like(x_start) 
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t =  t * z1 + (1.-t) * z0

        target = z1 - z0

        model_out = self.output(z_t, t.squeeze()*self.time_scalar, anomaly_label, None)
        train_loss = F.mse_loss(model_out, target, reduction='none')



        
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss.mean()
        return train_loss.mean()

    def _finetune_loss_on_normal(self, x_start, anomaly_label):

        z0 = torch.randn_like(x_start) # (B, T, dim)
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0 # (B, T, dim)

        target = z1 - z0

        model_out = self.output(z_t, t.squeeze() * self.time_scalar, anomaly_label, None)
        train_loss = F.mse_loss(model_out, target, reduction='none') # (B, T, dim)
        breakpoint()
        train_loss = train_loss * (1- anomaly_label)


        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss.mean()
        return train_loss.mean()

    def forward(self, x, anomaly_label):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        return self._train_loss(x, anomaly_label)

    def finetune_loss(self, x, anomaly_label, mode):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        assert mode in ("normal", "anomaly")
        if mode == "normal":
            return self._finetune_loss_on_normal(x, anomaly_label)
        elif mode == "anomaly":
            return self._train_loss(x, anomaly_label)
        else:
            raise ValueError("mode must be 'normal' or 'anomaly'")


    def fast_sample_infill(self, shape, target, partial_mask=None):

        z0 = torch.randn(shape).cuda()
        z1 = zt = z0
        for t in range(self.num_timesteps):
            t = t/self.num_timesteps  ## scale to 0-1
            t = t**(float(os.environ['hucfg_Kscale']))  ## perform t-power sampling

            
            z0 = torch.randn(shape).cuda()  ## re init the z0

            target_t = target*t + z0*(1-t)  ## get the noisy target
            zt = z1*t + z0*(1-t)  ##
            # import ipdb; ipdb.set_trace()
            zt[partial_mask] = target_t[partial_mask]  ## replace with the noisy version of given ground truth information
            v = self.output(zt, torch.tensor([t*self.time_scalar]).cuda(), None) 

            z1 = zt.clone() + (1 - t) * v  ## one step euler
            z1 = torch.clamp(z1, min=-1, max=1) ## make sure the upper and lower bound dont exceed


        return z1






