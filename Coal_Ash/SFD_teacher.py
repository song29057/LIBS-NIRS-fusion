import numpy as np
import torch
import torch.nn as nn
import random
import math
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, TensorDataset

# 设置环境和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

seed = 27 # 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class LIBSFeatureExtractor(nn.Module):
    def __init__(self):
        super(LIBSFeatureExtractor, self).__init__()
        self.ln1 = nn.LayerNorm(244)
        self.fc1 = nn.Linear(244, 64)
        self.rl1 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.rl1(x)
        return x


class LIBSSelfAttention(nn.Module):
    def __init__(self, dim):
        super(LIBSSelfAttention, self).__init__()
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


class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()

        self.extractor = LIBSFeatureExtractor()
        self.self_attention = LIBSSelfAttention(dim=64)

        # 输出层
        self.ln1 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 64)
        self.rl1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.rl2 = nn.ReLU()

        self.ln3 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.extractor(x)
        output = self.self_attention(x)

        output = self.ln1(output)
        output = self.fc1(output)
        output = self.rl1(output) + x

        output = self.ln2(output)
        output = self.fc2(output)
        feature = self.rl2(output)

        output = self.ln3(feature)
        output = self.fc3(output)

        output = output.squeeze(-1)

        return output, feature


class SpecData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return np.array(x), np.array(y)

    def __len__(self):
        return len(self.y)


XTrain_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/XTrain_LIBS_aug.csv', delimiter=',')
XValid_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/XValid_LIBS.csv', delimiter=',')
XTest_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/XTest_LIBS.csv', delimiter=',')
YTrain = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/YTrain_aug.csv', delimiter=',')
YValid = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/YValid.csv', delimiter=',')
YTest = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Coal Shandong/YTest.csv', delimiter=',')

# 数据标准化
scaler_x = StandardScaler().fit(XTrain_LIBS)
XTrain_LIBS = scaler_x.transform(XTrain_LIBS)
XValid_LIBS = scaler_x.transform(XValid_LIBS)
XTest_LIBS = scaler_x.transform(XTest_LIBS)

# 创建数据集实例
train_set = SpecData(XTrain_LIBS, YTrain)
valid_set = SpecData(XValid_LIBS, YValid)
test_set = SpecData(XTest_LIBS, YTest)

batch_size = 64
learning_rate = 0.0001
num_epochs = 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = TeacherModel().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 训练准备
best_val_loss = float('inf')
best_model_path = r'D:\Postdoc\Paper 12\Datasets\Coal Shandong\Models\teacher_model.pth'
train_losses = []
val_losses = []
best_epoch = 0

# 训练循环
print("开始训练...")
start_time = time.time()

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, feature = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item() * inputs.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证阶段
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, feature = model(inputs)
            running_val_loss += criterion(outputs, labels).item() * inputs.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    # 更新学习率
    scheduler.step()

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), best_model_path)

    # 打印进度
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'Train Loss: {train_loss:.4f} | '
          f'Val Loss: {val_loss:.4f} | '
          f'Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')

end_time = time.time()
print(f"训练完成，耗时: {(end_time - start_time) / 60:.2f} 分钟")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.show()

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"\n加载最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

# 评估验证集
val_true, val_pred = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, feature = model(inputs)
        val_true.extend(labels.cpu().numpy())
        val_pred.extend(outputs.cpu().numpy())

val_true = np.array(val_true)
val_pred = np.array(val_pred)
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
val_mae = mean_absolute_error(val_true, val_pred)
val_r2 = r2_score(val_true, val_pred)

print(f"\n验证集结果 (最佳模型): MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

# 评估测试集
test_true, test_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, feature = model(inputs)
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(outputs.cpu().numpy())

test_true = np.array(test_true)
test_pred = np.array(test_pred)
test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
test_mae = mean_absolute_error(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)

print(f"测试集结果 (最佳模型): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")

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
    x=max_val - 0.25*(max_val-min_val),  # 右侧留5%边距
    y=min_val + 0.05*(max_val-min_val),  # 底部留5%边距
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
