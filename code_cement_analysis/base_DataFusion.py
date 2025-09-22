import numpy as np
import torch
import torch.nn as nn
import random
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import csv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, TensorDataset

# 设置环境和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

seed = 27
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Input dimension is 380 (NIR) + 352 (LIBS) = 732
        self.ln1 = nn.LayerNorm(732)
        self.fc1 = nn.Linear(732, 128)  # Increased dimension to capture combined features
        self.rl1 = nn.ReLU()

        # Additional layer for better feature extraction
        self.ln2 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.rl2 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.rl1(x)

        x = self.ln2(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.self_attention = SelfAttention(dim=64)

        # Output layers
        self.ln1 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 64)
        self.rl1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.rl2 = nn.ReLU()

        self.ln3 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, nir_data, libs_data):
        # Combine the input data
        combined_input = torch.cat([nir_data, libs_data], dim=-1)

        # Extract features
        features = self.feature_extractor(combined_input)

        # Self-attention
        attn_output = self.self_attention(features)

        # Residual connection
        features = attn_output + features

        # Output layers
        output = self.ln1(features)
        output = self.fc1(output)
        output = self.rl1(output)

        output = self.ln2(output)
        output = self.fc2(output)
        output = self.rl2(output)

        output = self.ln3(output)
        output = self.fc3(output)

        output = output.squeeze(-1)

        return output


class SpecData(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1.astype(np.float32)
        self.x2 = x2.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y = self.y[idx]
        return torch.tensor(x1), torch.tensor(x2), torch.tensor(y)

    def __len__(self):
        return len(self.y)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): 在停止前等待的epoch数
            min_delta (float): 被视为改进的最小变化
            verbose (bool): 是否打印早停信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


# 加载数据
XTrain_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XTrain_LIBS_aug.csv', delimiter=',')
XValid_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XValid_LIBS.csv', delimiter=',')
XTest_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XTest_LIBS.csv', delimiter=',')
XTrain_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XTrain_NIRS_aug.csv', delimiter=',')
XValid_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XValid_NIRS.csv', delimiter=',')
XTest_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/XTest_NIRS.csv', delimiter=',')
YTrain = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/YTrain_aug.csv', delimiter=',')
YValid = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/YValid.csv', delimiter=',')
YTest = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Cement/YTest.csv', delimiter=',')

# 数据标准化
scaler_x1 = StandardScaler().fit(XTrain_NIRS)
XTrain_NIRS = scaler_x1.transform(XTrain_NIRS)
XValid_NIRS = scaler_x1.transform(XValid_NIRS)
XTest_NIRS = scaler_x1.transform(XTest_NIRS)

scaler_x2 = StandardScaler().fit(XTrain_LIBS)
XTrain_LIBS = scaler_x2.transform(XTrain_LIBS)
XValid_LIBS = scaler_x2.transform(XValid_LIBS)
XTest_LIBS = scaler_x2.transform(XTest_LIBS)

# 创建数据集实例
train_set = SpecData(XTrain_NIRS, XTrain_LIBS, YTrain)
valid_set = SpecData(XValid_NIRS, XValid_LIBS, YValid)
test_set = SpecData(XTest_NIRS, XTest_LIBS, YTest)

batch_size = 64
learning_rate = 0.001
num_epochs = 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = FusionModel().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
print(f"参数内存: {param_mem:.2f} KB")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 训练准备
best_val_loss = float('inf')
best_model_path = r'D:\Postdoc\Paper 12\Datasets\Cement\Models\DataFusion_model.pth'
train_losses = []
val_losses = []
best_epoch = 0

# 初始化早停
early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)

# Prepare CSV file path
csv_path = r'D:\Postdoc\Paper 12\Datasets\Cement\Models\training_log_DataFusion.csv'

# Write CSV header
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate', 'EarlyStop Counter'])

# 训练循环
print("开始训练...")
total_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # 训练阶段
    model.train()
    running_train_loss = 0.0
    train_batch_start = time.time()
    for inputs1, inputs2, labels in train_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs1.size(0)
    train_batch_time = time.time() - train_batch_start

    train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    val_start_time = time.time()
    with torch.no_grad():
        for inputs1, inputs2, labels in val_loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            outputs = model(inputs1, inputs2)
            running_val_loss += criterion(outputs, labels).item() * inputs1.size(0)
    val_time = time.time() - val_start_time

    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # 检查早停
    early_stopping(val_loss, epoch + 1)

    # Save metrics to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, val_loss, current_lr, early_stopping.counter])

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)

    epoch_time = time.time() - epoch_start_time

    # 打印进度
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss:.4f} | '
          f'Epoch Time: {epoch_time:.2f}s (Train: {train_batch_time:.2f}s, Val: {val_time:.2f}s) | '
          f'Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}) | '
          f'EarlyStop: {early_stopping.counter}/{early_stopping.patience}')

    # 检查是否早停
    if early_stopping.early_stop:
        print(f"\n早停触发！在 Epoch {epoch + 1} 停止训练")
        print(f"最佳模型在 Epoch {early_stopping.best_epoch}，验证损失: {early_stopping.best_loss:.4f}")
        break

end_time = time.time()
total_train_time = end_time - total_start_time
print(f"训练完成，总耗时: {total_train_time:.2f} 秒 ({total_train_time / 60:.2f} 分钟)")
print(f"平均每个epoch耗时: {total_train_time / (epoch + 1):.2f} 秒")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress with Early Stopping')
plt.legend()
plt.show()

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"\n加载最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

# 评估验证集
val_start_time = time.time()
val_true, val_pred = [], []
with torch.no_grad():
    for inputs1, inputs2, labels in val_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs = model(inputs1, inputs2)
        val_true.extend(labels.cpu().numpy())
        val_pred.extend(outputs.cpu().numpy())
val_time = time.time() - val_start_time

val_true = np.array(val_true)
val_pred = np.array(val_pred)
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
val_mae = mean_absolute_error(val_true, val_pred)
val_r2 = r2_score(val_true, val_pred)
file_path = r"D:\Postdoc\Paper 12\Datasets\Cement\Prediction\Fusion\DataFusion_val.csv"
np.savetxt(file_path, val_pred, delimiter=",")

print(f"\n验证集评估耗时: {val_time:.2f} 秒")
print(f"验证集结果 (最佳模型): MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

# 评估测试集
test_start_time = time.time()
test_true, test_pred = [], []
with torch.no_grad():
    for inputs1, inputs2, labels in test_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs = model(inputs1, inputs2)
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(outputs.cpu().numpy())
test_time = time.time() - test_start_time

test_true = np.array(test_true)
test_pred = np.array(test_pred)
test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
test_mae = mean_absolute_error(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)
file_path = r"D:\Postdoc\Paper 12\Datasets\Cement\Prediction\Fusion\DataFusion_test.csv"
np.savetxt(file_path, test_pred, delimiter=",")

print(f"测试集评估耗时: {test_time:.2f} 秒")
print(f"测试集结果 (最佳模型): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

# 最终时间汇总
print("\n" + "=" * 60)
print("时间统计汇总:")
print("=" * 60)
print(f"总训练时间: {total_train_time:.2f} 秒 ({total_train_time / 60:.2f} 分钟)")
print(f"训练epoch数: {epoch + 1}/{num_epochs}")
print(f"平均每个epoch时间: {total_train_time / (epoch + 1):.2f} 秒")
print(f"验证集评估时间: {val_time:.2f} 秒")
print(f"测试集评估时间: {test_time:.2f} 秒")
print(f"总程序运行时间: {(time.time() - total_start_time):.2f} 秒")
print(f"早停节省时间: 提前 {(num_epochs - epoch - 1)} 个epoch停止训练")
print("=" * 60)

plt.figure(figsize=(8, 6))

# 绘制散点图
val_scatter = plt.scatter(YValid, val_pred, alpha=0.6, color='royalblue', s=5, label='Validation')
test_scatter = plt.scatter(YTest, test_pred, alpha=0.6, color='crimson', s=5, label='Test')

# 绘制对角线
min_val = min(YTest.min(), YValid.min())
max_val = max(YTest.max(), YValid.max())
diagonal = plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

# 设置坐标轴范围
margin = 0.1 * (max_val - min_val)
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

# 添加评估指标文本
metrics_text = (
    f"Validation Metrics:\n"
    f"MAE = {val_mae:.4f}\n"
    f"RMSE = {val_rmse:.4f}\n"
    f"R2 = {val_r2:.4f}\n\n"
    f"Test Metrics:\n"
    f"MAE = {test_mae:.4f}\n"
    f"RMSE = {test_rmse:.4f}\n"
    f"R2 = {test_r2:.4f}"
)

plt.text(
    x=max_val - 0.25 * (max_val - min_val),  # 右侧留5%边距
    y=min_val + 0.05 * (max_val - min_val),  # 底部留5%边距
    s=metrics_text,
    bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round', pad=0.5),
    fontsize=11,
    ha='left',  # 文本右对齐
    va='bottom'  # 文本底部对齐
)

# 添加标签和图例
plt.xlabel('Measured Values', fontsize=12, labelpad=10)
plt.ylabel('Predicted Values', fontsize=12, labelpad=10)

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
# 调整布局
plt.tight_layout()
plt.show()