import numpy as np
import torch
import torch.nn as nn
import random
import math
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader, TensorDataset
from shap import PermutationExplainer

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


class NIRFeatureExtractor(nn.Module):
    def __init__(self):
        super(NIRFeatureExtractor, self).__init__()
        self.ln1 = nn.LayerNorm(215)
        self.fc1 = nn.Linear(215, 64)
        self.rl1 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.rl1(x)
        x = x.squeeze(-1)
        return x


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
        x = x.squeeze(-1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.dim = dim

        # LIBS to NIR attention projections
        self.libs_to_nir_query = nn.Linear(dim, dim)
        self.libs_to_nir_key = nn.Linear(dim, dim)
        self.libs_to_nir_value = nn.Linear(dim, dim)

        # NIR to LIBS attention projections
        self.nir_to_libs_query = nn.Linear(dim, dim)
        self.nir_to_libs_key = nn.Linear(dim, dim)
        self.nir_to_libs_value = nn.Linear(dim, dim)

    def forward(self, nir_feat, libs_feat):
        # LIBS to NIR attention
        Q_libs = self.libs_to_nir_query(libs_feat)
        K_nir = self.libs_to_nir_key(nir_feat)
        V_nir = self.libs_to_nir_value(nir_feat)

        attention_scores = torch.matmul(Q_libs, K_nir.transpose(-2, -1)) / math.sqrt(self.dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attn_output_libs_to_nir = torch.matmul(attention_weights, V_nir)

        # NIR to LIBS attention
        Q_nir = self.nir_to_libs_query(nir_feat)
        K_libs = self.nir_to_libs_key(libs_feat)
        V_libs = self.nir_to_libs_value(libs_feat)

        attention_scores = torch.matmul(Q_nir, K_libs.transpose(-2, -1)) / math.sqrt(self.dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attn_output_nir_to_libs = torch.matmul(attention_weights, V_libs)

        return attn_output_nir_to_libs, attn_output_libs_to_nir


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.nir_extractor = NIRFeatureExtractor()
        self.libs_extractor = LIBSFeatureExtractor()
        self.cross_attention = CrossAttention(dim=64)

        # 门控机制
        self.ln0 = nn.LayerNorm(128)
        self.gate_fc = nn.Linear(64 * 2, 64)  # 输入是两个特征的拼接，输出是门控权重
        self.sigmoid = nn.Sigmoid()

        # 输出层
        self.ln1 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(64, 64)
        self.rl1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.rl2 = nn.ReLU()

        self.ln3 = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 32)
        self.rl3 = nn.ReLU()

        self.ln4 = nn.LayerNorm(32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, nir_data, libs_data):
        # 提取特征
        nir_feat = self.nir_extractor(nir_data)
        libs_feat = self.libs_extractor(libs_data)

        # 交叉注意力
        cross_feat_nir_to_libs, cross_feat_libs_to_nir = self.cross_attention(nir_feat, libs_feat)

        output1 = self.ln1(cross_feat_nir_to_libs)
        output1 = self.fc1(output1)
        output1 = self.rl1(output1) + libs_feat

        output2 = self.ln2(cross_feat_libs_to_nir)
        output2 = self.fc2(output2)
        output2 = self.rl2(output2) + nir_feat

        combined_feat = torch.cat([output1, output2], dim=-1)  # 拼接特征
        combined_feat = self.ln0(combined_feat)
        gate = self.gate_fc(combined_feat)
        gate = self.sigmoid(gate)  # 计算门控权重

        fused_feat = gate * output1 + (1 - gate) * output2  # 加权融合

        output = self.ln3(fused_feat)
        output = self.fc3(output)
        output = self.rl3(output)

        output = self.ln4(output)
        output = self.fc4(output)

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


# 加载数据
XTrain_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XTrain_LIBS_aug.csv', delimiter=',')
XValid_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XValid_LIBS.csv', delimiter=',')
XTest_LIBS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XTest_LIBS.csv', delimiter=',')
XTrain_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XTrain_NIRS_aug.csv', delimiter=',')
XValid_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XValid_NIRS.csv', delimiter=',')
XTest_NIRS = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/XTest_NIRS.csv', delimiter=',')
YTrain = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/YTrain_aug.csv', delimiter=',')
YValid = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/YValid.csv', delimiter=',')
YTest = np.loadtxt(r'D:/Postdoc/Paper 12/Datasets/Coal Shandong/YTest.csv', delimiter=',')

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
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

# 加载最佳模型
model = FusionModel().to(device)
best_model_path = r'D:\Postdoc\Paper 12\Datasets\Coal Shandong\Models\best_CAFF_model.pth'
model.load_state_dict(torch.load(best_model_path))
model.eval()
print("最佳模型已加载")

# 评估验证集
val_true, val_pred = [], []
with torch.no_grad():
    for inputs1, inputs2, labels in val_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs = model(inputs1, inputs2)
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
    for inputs1, inputs2, labels in test_loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        outputs = model(inputs1, inputs2)
        test_true.extend(labels.cpu().numpy())
        test_pred.extend(outputs.cpu().numpy())

test_true = np.array(test_true)
test_pred = np.array(test_pred)
test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
test_mae = mean_absolute_error(test_true, test_pred)
test_r2 = r2_score(test_true, test_pred)

print(f"测试集结果 (最佳模型): MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")


def get_attention_matrix(model, dataloader):
    device = next(model.parameters()).device
    model.eval()

    all_C_nir_to_libs = []
    all_C_libs_to_nir = []

    with torch.no_grad():
        for inputs1, inputs2, _ in dataloader:  # inputs1: NIRS, inputs2: LIBS
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

            batch_size = inputs1.shape[0]

            # Step 1: 提取特征并记录激活状态
            # NIR 特征
            h_nir = model.nir_extractor.ln1(inputs1)
            h_nir = model.nir_extractor.fc1(h_nir)
            r_nir = (h_nir > 0).float()  # 直接在线性层后记录激活状态 [B,64]
            feat_nir = model.nir_extractor.rl1(h_nir)  # [B,64]

            # LIBS 特征
            h_libs = model.libs_extractor.ln1(inputs2)
            h_libs = model.libs_extractor.fc1(h_libs)
            r_libs = (h_libs > 0).float()  # [B,64]
            feat_libs = model.libs_extractor.rl1(h_libs)  # [B,64]

            # Step 2: 获取权重矩阵 (保持原始代码的转置方式)
            W_nirs = model.nir_extractor.fc1.weight.T  # (380, 64)
            W_libs = model.libs_extractor.fc1.weight.T  # (352, 64)

            # Step 3: 计算原始注意力矩阵 (移除softmax和缩放)
            cross_attention = model.cross_attention
            Q_nir = cross_attention.nir_to_libs_query(feat_nir)  # (B, 64)
            K_libs = cross_attention.nir_to_libs_key(feat_libs)  # (B, 64)
            A_nir_to_libs = torch.bmm(Q_nir.unsqueeze(2), K_libs.unsqueeze(1)) / math.sqrt(64)  # [B,64,64]
            # A_nir_to_libs = F.softmax(A_nir_to_libs, dim=-1)  # 应用softmax

            Q_libs = cross_attention.libs_to_nir_query(feat_libs)
            K_nir = cross_attention.libs_to_nir_key(feat_nir)
            A_libs_to_nir = torch.bmm(Q_libs.unsqueeze(2), K_nir.unsqueeze(1)) / math.sqrt(64)  # [B,64,64]
            # A_libs_to_nir = F.softmax(A_libs_to_nir, dim=-1)  # 应用softmax

            # Step 4: 逐个样本计算相关性矩阵
            for i in range(batch_size):
                # 当前样本的掩码对角矩阵
                D_R_nirs = torch.diag(r_nir[i])  # (64, 64)
                D_R_libs = torch.diag(r_libs[i])  # (64, 64)

                # 当前注意力矩阵
                A_i_nir_to_libs = A_nir_to_libs[i]  # (64, 64)
                A_i_libs_to_nir = A_libs_to_nir[i]  # (64, 64)

                # 计算相关性矩阵 (保持原始公式)
                C_nir_to_libs = W_nirs @ D_R_nirs @ A_i_nir_to_libs @ D_R_libs @ W_libs.T
                C_libs_to_nir = W_libs @ D_R_libs @ A_i_libs_to_nir @ D_R_nirs @ W_nirs.T

                all_C_nir_to_libs.append(C_nir_to_libs.cpu().numpy())
                all_C_libs_to_nir.append(C_libs_to_nir.cpu().numpy())

    # Step 5: 取平均
    avg_C_nir_to_libs = np.mean(all_C_nir_to_libs, axis=0)  # (380, 352)
    avg_C_libs_to_nir = np.mean(all_C_libs_to_nir, axis=0)  # (352, 380)

    return {
        "nir_to_libs": avg_C_nir_to_libs,
        "libs_to_nir": avg_C_libs_to_nir
    }


result = get_attention_matrix(model, test_loader)
avg_C_nir_to_libs = result["nir_to_libs"]  # shape: (380, 352)
avg_C_libs_to_nir = result["libs_to_nir"]  # shape: (352, 380)

save_path = r'D:\Postdoc\Paper 12\Datasets\Coal Shandong\Interpretation'
np.savetxt(os.path.join(save_path, 'avg_C_nir_to_libs.csv'), avg_C_nir_to_libs, delimiter=',')
np.savetxt(os.path.join(save_path, 'avg_C_libs_to_nir.csv'), avg_C_libs_to_nir, delimiter=',')


import seaborn as sns
import matplotlib.pyplot as plt

# 假设 avg_C_libs_to_nir 是你已经计算好的相关性矩阵 (380 x 352)
vmin = avg_C_libs_to_nir.min()
vmax = avg_C_libs_to_nir.max()

# 使用 diverging 配色方案：中心为白色，正为红，负为蓝
cmap = "coolwarm"

plt.figure(figsize=(10, 8))
sns.heatmap(
    avg_C_nir_to_libs,
    cmap=cmap,
    center=0,               # 中心为 0，确保 0 显示为白色
    vmin=vmin,              # 最小值
    vmax=vmax,              # 最大值
    cbar=True,              # 显示颜色条
    xticklabels=50,         # 控制 LIBS 特征标签密度
    yticklabels=50          # 控制 NIR 波长标签密度
)

plt.title("NIR to LIBS Feature Correlation", fontsize=14)
plt.xlabel("LIBS Features", fontsize=12)
plt.ylabel("NIR Wavelengths", fontsize=12)

plt.tight_layout()
plt.show()