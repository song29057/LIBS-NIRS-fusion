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


class LIBSFeatureExtractor(nn.Module):
    def __init__(self):
        super(LIBSFeatureExtractor, self).__init__()
        self.ln1 = nn.LayerNorm(352)
        self.fc1 = nn.Linear(352, 64)
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
        feature = self.fc2(output)
        output = self.rl2(feature)

        output = self.ln3(output)
        output = self.fc3(output)

        output = output.squeeze(-1)

        return output, feature


######################################
class NIRSFeatureExtractor(nn.Module):
    def __init__(self):
        super(NIRSFeatureExtractor, self).__init__()
        self.ln1 = nn.LayerNorm(380)
        self.fc1 = nn.Linear(380, 64)
        self.rl1 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.rl1(x)
        return x


class NIRSSelfAttention(nn.Module):
    def __init__(self, dim):
        super(NIRSSelfAttention, self).__init__()
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


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()

        self.extractor = NIRSFeatureExtractor()
        self.self_attention = NIRSSelfAttention(dim=64)

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
        feature = self.fc2(output)
        output = self.rl2(feature)

        output = self.ln3(output)
        output = self.fc3(output)

        output = output.squeeze(-1)

        return output, feature


#########################################
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
learning_rate = 0.0005
num_epochs = 100

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# ====================== 加载预训练教师模型 ==================
# 定义教师模型的路径
teacher_model_path = r'D:\Postdoc\Paper 12\Datasets\Cement\Models\teacher_model.pth'

# 加载预训练教师模型
teacher_model = TeacherModel().to(device)
teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))

# 冻结教师模型的参数
for param in teacher_model.parameters():
    param.requires_grad = False

# ====================== 初始化学生模型 ======================
student_model = StudentModel().to(device)

total_params = sum(p.numel() for p in student_model.parameters())
print(f"总参数量: {total_params:,}")
param_mem = sum(p.numel() * p.element_size() for p in student_model.parameters()) / 1024
print(f"参数内存: {param_mem:.2f} KB")

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# 训练准备
best_val_loss = float('inf')
best_model_path = r'D:\Postdoc\Paper 12\Datasets\Cement\Models\student_model.pth'
train_losses = []
val_losses = []
best_epoch = 0

# Prepare CSV file path
csv_path = r'D:\Postdoc\Paper 12\Datasets\Cement\Models\training_log_KD.csv'

# Write CSV header
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])

train_mse_losses = []
train_cosine_losses = []
val_mse_losses = []
val_cosine_losses = []

# 训练循环
print("开始训练...")
start_time = time.time()

for epoch in range(num_epochs):
    # 训练阶段
    teacher_model.eval()
    student_model.train()
    running_train_loss = 0.0
    running_train_mse = 0.0
    running_train_cosine = 0.0

    for nir_data, libs_data, labels in train_loader:
        nir_data, libs_data, labels = nir_data.to(device), libs_data.to(device), labels.to(device)
        with torch.no_grad():
            teacher_output, teacher_feature = teacher_model(libs_data)
        optimizer.zero_grad()
        student_output, student_feature = student_model(nir_data)

        mse_loss = criterion(student_output, labels)
        cosine_loss = 1 - F.cosine_similarity(student_feature, teacher_feature).mean()
        loss = 0.7 * mse_loss + 0.3 * cosine_loss  # 总损失

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * nir_data.size(0)
        running_train_mse += mse_loss.item() * nir_data.size(0)
        running_train_cosine += cosine_loss.item() * nir_data.size(0)

    train_loss = running_train_loss / len(train_loader.dataset)
    train_mse = running_train_mse / len(train_loader.dataset)
    train_cosine = running_train_cosine / len(train_loader.dataset)

    train_losses.append(train_loss)
    train_mse_losses.append(train_mse)
    train_cosine_losses.append(train_cosine)

    # 验证阶段
    student_model.eval()
    running_val_loss = 0.0
    running_val_mse = 0.0
    running_val_cosine = 0.0

    with torch.no_grad():
        for nir_data, libs_data, labels in val_loader:
            nir_data, libs_data, labels = nir_data.to(device), libs_data.to(device), labels.to(device)
            teacher_output, teacher_feature = teacher_model(libs_data)
            student_output, student_feature = student_model(nir_data)

            mse_loss = criterion(student_output, labels)
            cosine_loss = 1 - F.cosine_similarity(student_feature, teacher_feature).mean()
            loss = 0.7 * mse_loss + 0.3 * cosine_loss  # 总损失

            running_val_loss += loss.item() * nir_data.size(0)
            running_val_mse += mse_loss.item() * nir_data.size(0)
            running_val_cosine += cosine_loss.item() * nir_data.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    val_mse = running_val_mse / len(val_loader.dataset)
    val_cosine = running_val_cosine / len(val_loader.dataset)

    val_losses.append(val_loss)
    val_mse_losses.append(val_mse)
    val_cosine_losses.append(val_cosine)

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Save metrics to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, train_loss, val_loss, current_lr])

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(student_model.state_dict(), best_model_path)

    # 打印进度
    print(f'Epoch [{epoch + 1}/{num_epochs}] | '
          f'Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Cos: {train_cosine:.4f}) | '
          f'Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Cos: {val_cosine:.4f}) | '
          f'Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')

end_time = time.time()
print(f"训练完成，耗时: {(end_time - start_time) / 60:.2f} 分钟")

# 绘制损失曲线
plt.figure(figsize=(15, 10))

# 训练集损失曲线
plt.subplot(2, 2, 1)
plt.plot(train_mse_losses, label='Train MSE Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training MSE Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_cosine_losses, label='Train Cosine Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Cosine Similarity Loss')
plt.legend()

# 验证集损失曲线
plt.subplot(2, 2, 3)
plt.plot(val_mse_losses, label='Validation MSE Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation MSE Loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(val_cosine_losses, label='Validation Cosine Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Cosine Similarity Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制总损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Total Loss')
plt.plot(val_losses, label='Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Training Progress (0.7*MSE + 0.3*Cosine)')
plt.legend()
plt.show()


# 加载最佳模型
student_model.load_state_dict(torch.load(best_model_path))
student_model.eval()
print(f"\n加载最佳模型 (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

# 评估验证集
val_true, val_pred = [], []
with torch.no_grad():
    for nir_data, libs_data, labels in val_loader:
        nir_data, labels = nir_data.to(device), labels.to(device)
        outputs, feature = student_model(nir_data)
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
    for nir_data, libs_data, labels in test_loader:
        nir_data, labels = nir_data.to(device), labels.to(device)
        outputs, feature = student_model(nir_data)
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


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
#
# # 提取特征
# teacher_features = []
# student_features = []
# with torch.no_grad():
#     for nir_data, libs_data, _ in test_loader:
#         _, t_feat = teacher_model(libs_data.to(device))
#         _, s_feat = student_model(nir_data.to(device))
#         teacher_features.append(t_feat.cpu())
#         student_features.append(s_feat.cpu())
#
# teacher_features = torch.cat(teacher_features).numpy()
# student_features = torch.cat(student_features).numpy()
#
# # t-SNE降维
# tsne = TSNE(n_components=2, random_state=42)
# t_tsne = tsne.fit_transform(teacher_features)
# s_tsne = tsne.fit_transform(student_features)
#
# # 绘制对比图
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.scatter(t_tsne[:,0], t_tsne[:,1], c=YTest, cmap='viridis', alpha=0.6)
# plt.title("Teacher Feature Space (LIBS)")
# plt.colorbar(label='Target Value')
#
# plt.subplot(122)
# plt.scatter(s_tsne[:,0], s_tsne[:,1], c=YTest, cmap='viridis', alpha=0.6)
# plt.title("Student Feature Space (NIR)")
# plt.colorbar(label='Target Value')
# plt.show()