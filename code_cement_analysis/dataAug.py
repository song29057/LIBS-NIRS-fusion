import numpy as np
import torch
import torch.nn as nn
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from sklearn.utils import shuffle
import torch.nn.functional as F
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


XTrain_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XTrain_LIBS.csv', delimiter=',')
XTrain_NIRS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XTrain_NIRS.csv', delimiter=',')
XValid_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XValid_LIBS.csv', delimiter=',')
XValid_NIRS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XValid_NIRS.csv', delimiter=',')
XTest_LIBS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XTest_LIBS.csv', delimiter=',')
XTest_NIRS = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/XTest_NIRS.csv', delimiter=',')
YTrain = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/YTrain.csv', delimiter=',')
YValid = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/YValid.csv', delimiter=',')
YTest = np.loadtxt(r'D:\Postdoc\Paper 12\Datasets\Cement/YTest.csv', delimiter=',')


def augment_data(x1, x2, y, num_augment=14, noise_ratio=0.05, min_noise_std=1e-5, max_noise_std=1e-3,
                 shuffle_data=True):
    """
    改进的数据增强函数

    参数:
        x1, x2: 输入特征数据
        y: 标签数据
        num_augment: 每个样本的增强次数
        noise_ratio: 噪声比例
        min_noise_std: 最小噪声标准差
        max_noise_std: 最大噪声标准差
        shuffle_data: 是否打乱数据

    返回:
        增强后的数据
    """
    augmented_x1 = [x1.copy()]
    augmented_x2 = [x2.copy()]
    augmented_y = [y.copy()]

    for _ in range(num_augment):
        # 计算噪声标准差 - 使用全局统计量而不是单个样本
        std_x1 = np.std(x1, axis=0)
        std_x2 = np.std(x2, axis=0)

        noise_std_x1 = np.clip(std_x1 * noise_ratio, min_noise_std, max_noise_std)
        noise_std_x2 = np.clip(std_x2 * noise_ratio, min_noise_std, max_noise_std)

        # 生成噪声
        noise_x1 = np.random.normal(0, noise_std_x1, size=x1.shape)
        noise_x2 = np.random.normal(0, noise_std_x2, size=x2.shape)

        # 添加噪声
        augmented_x1.append(x1 + noise_x1)
        augmented_x2.append(x2 + noise_x2)
        augmented_y.append(y.copy())

    # 合并所有增强数据
    augmented_x1 = np.concatenate(augmented_x1, axis=0)
    augmented_x2 = np.concatenate(augmented_x2, axis=0)
    augmented_y = np.concatenate(augmented_y, axis=0)

    # 打乱数据
    if shuffle_data:
        augmented_x1, augmented_x2, augmented_y = shuffle(augmented_x1, augmented_x2, augmented_y)

    return augmented_x1, augmented_x2, augmented_y


# 使用示例
XTrain_LIBS_aug, XTrain_NIRS_aug, YTrain_aug = augment_data(
    XTrain_LIBS,
    XTrain_NIRS,
    YTrain,
    num_augment=19
)


# 加载数据


# scaler_x1 = StandardScaler().fit(XTrain_LIBS)
# XTrain_LIBS = scaler_x1.transform(XTrain_LIBS)
# XValid_LIBS = scaler_x1.transform(XValid_LIBS)
# XTest_LIBS = scaler_x1.transform(XTest_LIBS)
#
# scaler_x2 = StandardScaler().fit(XTrain_NIRS)
# XTrain_NIRS = scaler_x2.transform(XTrain_NIRS)
# XValid_NIRS = scaler_x2.transform(XValid_NIRS)
# XTest_NIRS = scaler_x2.transform(XTest_NIRS)

# 数据增强
# XTrain_NIRS_aug, XTrain_LIBS_aug, YTrain_aug = augment_data(XTrain_NIRS, XTrain_LIBS, YTrain, num_augment=14)
print(YTrain_aug.shape)


# 绘制原始数据和增强后的数据
def plot_spectra(original_data, augmented_data, title):
    plt.figure(figsize=(10, 6))
    for spectrum in original_data:
        plt.plot(spectrum, color='red', alpha=0.5)
    for spectrum in augmented_data:
        plt.plot(spectrum, color='blue', alpha=0.2)
    plt.title(title)
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.show()
# 绘制数据


plot_spectra(XTrain_NIRS_aug, XTrain_NIRS, 'NIRS Spectra: Augmented (Red) vs Original (Blue)')
plot_spectra(XTrain_LIBS_aug, XTrain_LIBS, 'LIBS Spectra: Augmented (Red) vs Original (Blue)')

# 定义保存数据的路径
save_path = r'D:\Postdoc\Paper 12\Datasets\Cement'

# 确保保存目录存在
os.makedirs(save_path, exist_ok=True)

# 保存增强后的数据
np.savetxt(os.path.join(save_path, 'XTrain_NIRS_aug.csv'), XTrain_NIRS_aug, delimiter=',')
np.savetxt(os.path.join(save_path, 'XTrain_LIBS_aug.csv'), XTrain_LIBS_aug, delimiter=',')
np.savetxt(os.path.join(save_path, 'YTrain_aug.csv'), YTrain_aug, delimiter=',')

print("增强后的数据已成功保存到指定路径")