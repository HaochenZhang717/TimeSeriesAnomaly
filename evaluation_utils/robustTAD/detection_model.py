import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


# -----------------------------
# 基础模块：Double Conv (Conv → ReLU → Conv → ReLU)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 上采样模块：UpConv + DoubleConv
# -----------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # 修正长度不一致问题（常见）
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (0, diff))

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -----------------------------
# 主网络：UNet 时间序列异常检测
# -----------------------------
class UNetTS(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        # Encoder: 通道和图一致
        self.conv1 = DoubleConv(in_ch, 16)    # 240
        self.conv2 = DoubleConv(16, 32)       # 120
        self.conv3 = DoubleConv(32, 64)       # 60
        self.conv4 = DoubleConv(64, 128)      # 30
        self.conv5 = DoubleConv(128, 256)     # 15

        self.pool = nn.MaxPool1d(2)

        # Decoder
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        self.up1 = UpBlock(32, 16)

        # Final output → per-time step logits
        self.final = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # ---- Encoder ----
        c1 = self.conv1(x)        # (B,16,240)
        p1 = self.pool(c1)        # (B,16,120)

        c2 = self.conv2(p1)       # (B,32,120)
        p2 = self.pool(c2)        # (B,32,60)

        c3 = self.conv3(p2)       # (B,64,60)
        p3 = self.pool(c3)        # (B,64,30)

        c4 = self.conv4(p3)       # (B,128,30)
        p4 = self.pool(c4)        # (B,128,15)

        c5 = self.conv5(p4)       # (B,256,15)

        # ---- Decoder ----
        u4 = self.up4(c5, c4)     # (B,128,30)
        u3 = self.up3(u4, c3)     # (B,64,60)
        u2 = self.up2(u3, c2)     # (B,32,120)
        u1 = self.up1(u2, c1)     # (B,16,240)

        logits = self.final(u1)      # (B,2,240)
        return logits


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, beta=1.0):
        """
        使用BCEWithLogitsLoss，它在内部处理logits，数值更稳定
        pos_weight: 正样本权重 = beta
        """
        super().__init__()
        self.beta = beta

    def forward(self, logits, labels):
        """
        logits: (B,1,T)
        labels: (B,T)
        """
        # 挤压掉通道维度，BCEWithLogitsLoss期望(B,T)或(B,C,T)
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # (B,T)

        # pos_weight = beta 表示：正样本的损失乘以beta
        # 例如：beta=5，那么每个正样本的损失贡献是负样本的5倍
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.beta], device=logits.device),
            reduction='mean'
        )
        return loss_fn(logits, labels.float())


class RobustTAD(nn.Module):
    def __init__(self, in_ch, anomaly_weight):
        super().__init__()
        self.model = UNetTS(in_ch=in_ch)
        self.criterion = WeightedBCEWithLogitsLoss(beta=anomaly_weight)

    def forward(self, inputs, labels):
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        return loss

    def predict(self, inputs):
        logits = self.model(inputs)
        probs = torch.sigmoid(logits)
        return (probs > 0.5).to(torch.long).squeeze(1)



def calculate_robustTAD(
        anomaly_weight, feature_size,
        ori_data, ori_labels,
        gen_data, gen_labels,
        device, lr,
        max_epochs=2000,
        batch_size=64,
        patience=20,
):
    X_real = torch.tensor(ori_data, dtype=torch.float32)
    X_fake = torch.tensor(gen_data, dtype=torch.float32)

    y_real = torch.tensor(ori_labels, dtype=torch.float32)
    y_fake = torch.tensor(gen_labels, dtype=torch.float32)

    train_ds = TensorDataset(X_fake, y_fake)
    test_ds = TensorDataset(X_real, y_real)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = RobustTAD(in_ch=feature_size, anomaly_weight=anomaly_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # multiply LR by 0.5
        patience=2,  # wait 3 epochs with no improvement
        threshold=1e-4,  # improvement threshold
        min_lr=1e-6,  # min LR clamp
    )



    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0


    for epoch in range(max_epochs):
        model.train()
        # for Xb, yb in tqdm(train_loader, desc=f"Epoch{epoch}"):
        train_loss = 0.0
        train_seen = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = model(Xb, yb)
            # breakpoint()
            # print(loss.item())
            train_loss += loss.item() * Xb.shape[0]
            train_seen += Xb.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss_avg = train_loss / train_seen
        model.eval()
        val_loss = 0.0
        val_seen = 0
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            with torch.no_grad():
                loss = model(Xb, yb)
            val_loss += loss.item() * Xb.shape[0]
            val_seen += Xb.shape[0]
        val_loss_avg = val_loss / val_seen
        print(f"Epoch{epoch}: train_loss: {train_loss_avg} | val_loss: {val_loss_avg} | lr: {optimizer.param_groups[0]['lr']} ||")
        scheduler.step(val_loss_avg)

        if best_val_loss > val_loss_avg:
            best_val_loss = val_loss_avg
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val_loss = {best_val_loss:.6f}")
            break

    model.load_state_dict(best_state)
    model.eval()

    ### run evaluation on test set
    normal_correct = 0
    normal_num = 0
    anomaly_correct = 0
    anomaly_num = 0
    all_preds = []
    all_labels = []

    for Xb, yb in test_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        y_pred = model.predict(Xb)
        normal_num += (yb == 0).sum().item()
        anomaly_num += (yb == 1).sum().item()
        normal_correct += ((y_pred==yb) * (yb==0)).sum().item()
        anomaly_correct += ((y_pred==yb) * (yb==1)).sum().item()
        all_preds.append(y_pred.detach().cpu())
        all_labels.append(yb.detach().cpu())
    normal_accuracy = normal_correct / normal_num
    anomaly_accuracy = anomaly_correct / anomaly_num

    all_preds = torch.cat(all_preds).flatten().numpy()
    all_labels = torch.cat(all_labels).flatten().numpy()

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return normal_accuracy, anomaly_accuracy, precision, recall, f1




if __name__ == '__main__':
    model = UNetTS(in_ch=2, num_classes=2)

    x = torch.randn(4, 800, 2)  # batch=4
    logits = model(x)

    print(logits.shape)