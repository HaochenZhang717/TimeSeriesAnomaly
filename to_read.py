def _impute_loss(self, x_start, anomaly_label):
        # x_start == [1,2,3,4,5,6]
        # anomaly_label = [0,0,1,1,1,0]
        z0_impute = torch.randn_like(x_start) * anomaly_label.unsqueeze(-1) + x_start * (1 - anomaly_label.unsqueeze(-1)) #[1,2,noise,noise,noise,6]
        z1 = x_start # [1,2,3,4,5,6]

        t = torch.rand(z0_impute.shape[0], 1, 1).to(z0_impute.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0_impute.shape[0], 1, 1)).to(z0_impute.device)

        z_t = t * z1 + (1. - t) * z0_impute # [1,2,3+noise,4+noise,5+noise,6]

        target = (z1 - z0_impute) * anomaly_label.unsqueeze(-1) # [0,0,3-noise, 4-noise, 5-noise, 0]
        model_out = self.output(z_t, t.view(-1) * self.time_scalar, None)
        model_out = model_out * anomaly_label.unsqueeze(-1)

        # train_loss: (B, ..., ...)
        train_loss = ((model_out - target) ** 2).mean(-1) #(B, T)
        # 只对 anomaly 部分计算误差
        masked_loss = train_loss * anomaly_label #(B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(anomaly_label, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()