"""
losses.py - Fluid LOD 预测损失函数库
=====================================
包含: Sobolev Loss, 增强物理 Loss, 加权 MSE 等。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LastDayLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_val = self.mse(pred, target)
        pred_last = pred[:, -1] - pred[:, -1]
        target_last = target[:, -1] - target[:, -1]
        loss_last = self.mse(pred_last, target_last)
        return loss_val + self.alpha * loss_last


class PhysicsLoss(nn.Module):
    """
    Sobolev Loss (数值 + 一阶差分)
    强制模型同时预测准值和变化趋势。
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_val = self.mse(pred, target)
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        loss_diff = self.mse(pred_diff, target_diff)
        return loss_val + self.alpha * loss_diff

class WeightedMSELoss(nn.Module):
    """
    按预报步长加权 MSE
    scheme: 'linear', 'exponential', 'last_day'
    """

    def __init__(self, scheme='linear', start_weight=1.0, end_weight=3.0):
        super().__init__()
        self.scheme = scheme
        self.start_weight = start_weight
        self.end_weight = end_weight

    def forward(self, pred, target):
        T = pred.shape[1]
        device = pred.device

        if self.scheme == 'linear':
            w = torch.linspace(self.start_weight, self.end_weight, T, device=device)
        elif self.scheme == 'exponential':
            w = torch.exp(torch.linspace(0, 1, T, device=device))
            w = w / w.mean() * (self.start_weight + self.end_weight) / 2
        elif self.scheme == 'last_day':
            w = torch.ones(T, device=device) * 0.1
            w[-1] = 10.0
        else:
            w = torch.ones(T, device=device)

        return (w * (pred - target).pow(2)).mean()


class HuberPhysicsLoss(nn.Module):
    """Huber Loss + 差分约束 (对异常值更鲁棒)"""

    def __init__(self, delta=1.0, alpha=0.5):
        super().__init__()
        self.delta = delta
        self.alpha = alpha

    def forward(self, pred, target):
        loss_val = F.huber_loss(pred, target, delta=self.delta)
        pred_d = pred[:, 1:] - pred[:, :-1]
        target_d = target[:, 1:] - target[:, :-1]
        loss_d = F.huber_loss(pred_d, target_d, delta=self.delta)
        return loss_val + self.alpha * loss_d


class KoopmanLoss(nn.Module):
    """
    带有严格物理稳态约束的 Deep Koopman 损失函数
    """

    def __init__(self, alpha=1.0, beta=0.1, gamma=5.0):
        super().__init__()
        self.alpha = alpha  # 线性转移损失权重
        self.beta = beta  # K 矩阵 L2 正则权重
        self.gamma = gamma  # 特征值越界惩罚权重 (极高)
        self.mse_loss = nn.MSELoss()

    def forward(self, final_pred, target, z_history, K_matrix):
        # 1. 预测损失 (MSE)
        loss_pred = self.mse_loss(final_pred, target)

        # 2. Koopman 线性转移损失
        z_t = z_history[:, :-1, :]
        z_t_plus_1 = z_history[:, 1:, :]
        z_t_next_pred = torch.matmul(z_t, K_matrix.t())
        loss_transition = self.mse_loss(z_t_next_pred, z_t_plus_1)

        # 3. L2 正则，防止矩阵元素绝对值过大
        loss_reg = torch.mean(K_matrix ** 2)

        # 4. 🚨 核心救命稻草：特征值单位圆约束 (Eigenvalue Penalty)
        # 计算 K 矩阵的所有复数特征值
        eigenvalues = torch.linalg.eigvals(K_matrix)
        # 计算特征值的模长 (绝对值)
        abs_eigen = torch.abs(eigenvalues)
        # 我们只惩罚那些模长大于 0.99 的特征值 (强制系统处于收敛或稳定振荡状态)
        # 用 ReLU 截断，小于 0.99 的不惩罚
        loss_eig = torch.mean(torch.relu(abs_eigen - 0.99))

        # 5. 总损失
        total_loss = loss_pred + self.alpha * loss_transition + self.beta * loss_reg + self.gamma * loss_eig

        return total_loss, loss_pred, loss_transition