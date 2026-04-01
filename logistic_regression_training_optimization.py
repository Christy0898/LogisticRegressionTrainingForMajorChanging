# -*- coding: utf-8 -*-
"""
模型训练与优化实现（对应论文 3.2.2 - 3.2.4）
使用全特征模型，不进行特征筛选。
数据集: data_for_logistic_regression.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.base import BaseEstimator, ClassifierMixin

# ==================== 1. 数据加载与划分 ====================
data = pd.read_csv('data_for_logistic_regression.csv')
feature_cols = [c for c in data.columns if c != 'Y']   # 获取特征列名
X = data[feature_cols].values
y = data['Y'].values

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征列: {feature_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"训练集样本量: {len(X_train)}, 测试集样本量: {len(X_test)}")
print(f"训练集高风险比例: {np.mean(y_train):.2%}")
print(f"测试集高风险比例: {np.mean(y_test):.2%}")

# ==================== 2. 自定义带 L2 正则化的逻辑回归（梯度下降） ====================
class LogisticRegressionL2(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, learning_rate=0.03, n_iter=1000, lambda_reg=0.1, verbose=True):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.weights = None
        self.loss_history = []

    def sigmoid(self, z):
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X_aug = np.c_[np.ones(X.shape[0]), X]   # 添加截距列
        n, p = X_aug.shape
        # 初始化权重（小随机数，避免初始梯度爆炸）
        self.weights = np.random.randn(p) * 0.01
        # 梯度下降
        for i in range(self.n_iter):
            z = X_aug @ self.weights
            y_pred = self.sigmoid(z)
            # 损失（加极小值防止log(0)）
            loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
            reg = 0.5 * self.lambda_reg * np.sum(self.weights[1:] ** 2)
            total_loss = loss + reg
            self.loss_history.append(total_loss)
            # 梯度（带 L2 正则化）
            grad = (1 / n) * X_aug.T @ (y_pred - y)
            grad[1:] += self.lambda_reg * self.weights[1:]
            # 更新参数
            self.weights -= self.learning_rate * grad
            # 打印
            if self.verbose and (i % 100 == 0 or i == self.n_iter - 1):
                print(f"Iter {i:4d}: loss = {total_loss:.6f}")

        # sklearn 兼容属性，供 check_is_fitted 和 API 预期
        self.coef_ = self.weights[1:].reshape(1, -1)
        self.intercept_ = np.array([self.weights[0]])
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X_aug = np.c_[np.ones(X.shape[0]), X]
        z = X_aug @ self.weights
        p1 = self.sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            p1 = proba[:, 1]
        else:
            p1 = proba.ravel()
        return (p1 >= threshold).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ==================== 3. 模型训练（全特征） ====================
print("\n" + "=" * 50)
print("全特征模型训练（L2正则化）")
print("=" * 50)
model = LogisticRegressionL2(learning_rate=0.03, n_iter=1000,
                             lambda_reg=0.1, verbose=True)
model.fit(X_train, y_train)

# 先定位核心特征 WR 和 xC 的索引
try:
    WR_idx = feature_cols.index('WR')
    xC_idx = feature_cols.index('xC')
except ValueError:
    WR_idx = 0
    xC_idx = 3
    print(f"警告：未找到 'WR' 或 'xC'，使用默认索引 {WR_idx}, {xC_idx}")

print("\n模型系数（前几个特征）:")
print(f"WR 系数: {model.weights[1+WR_idx]:.6f}")  # 因为 weights[0] 是截距
print(f"xC 系数: {model.weights[1+xC_idx]:.6f}")

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\n准确率 - 训练集: {train_acc:.4f}, 测试集: {test_acc:.4f}")

# 损失收敛曲线
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('损失收敛曲线')
plt.grid(True)
plt.savefig('loss_curve.png', dpi=150)
plt.show()

# ==================== 4. 模型评估 ====================
y_pred_proba = model.predict_proba(X_test)
if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
    y_pred_proba = y_pred_proba[:, 1]
else:
    y_pred_proba = y_pred_proba.ravel()
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n测试集 AUC: {auc:.4f}")

# ==================== 5. 保存模型参数为 CSV ====================
params_data = [['Intercept', model.weights[0]]]
for i, coef in enumerate(model.weights[1:]):
    params_data.append([f'{feature_cols[i]}', coef, i])

df_params = pd.DataFrame(params_data, columns=['Feature', 'Coefficient', 'Original_Index'])
df_params.to_csv('model_full_features.csv', index=False)
print("\n全特征模型参数已保存至 model_full_features.csv")
print("文件内容预览：")
print(df_params)

# 同时保存为 NPZ
np.savez('model_full_features.npz',
         weights=model.weights,
         feature_count=len(feature_cols))
print("全特征模型参数同时保存至 model_full_features.npz")

# ==================== 6. 可视化结果（增强版） ====================
print("\n" + "=" * 50)
print("生成可视化结果：决策边界、预测概率分布、校准曲线")
print("=" * 50)

# 6.0 预测概率分布诊断
y_pred_proba_all = model.predict_proba(X)
if y_pred_proba_all.ndim == 2 and y_pred_proba_all.shape[1] == 2:
    y_pred_proba_all = y_pred_proba_all[:, 1]
else:
    y_pred_proba_all = y_pred_proba_all.ravel()
plt.figure(figsize=(6, 4))
plt.hist(y_pred_proba_all, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prob_distribution.png', dpi=300)
plt.show()

# 6.1 决策边界图（使用核心特征：WR 和 xC）
# 获取特征索引

WR_idx = feature_cols.index('WR') if "WR" in feature_cols else 0
xC_idx = feature_cols.index('xC') if "xC" in feature_cols else 3

X_WR = X[:, WR_idx]
X_xC = X[:, xC_idx]

# 坐标轴范围：取数据 2% 和 98% 分位数，避免离群点拉伸
x_min = np.percentile(X_WR, 2)
x_max = np.percentile(X_WR, 98)
y_min = np.percentile(X_xC, 2)
y_max = np.percentile(X_xC, 98)

x_pad = (x_max - x_min) * 0.05
y_pad = (y_max - y_min) * 0.05
x_min, x_max = x_min - x_pad, x_max + x_pad
y_min, y_max = y_min - y_pad, y_max + y_pad

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

X_grid = np.zeros((xx.ravel().shape[0], X.shape[1]))

X_grid[:, WR_idx] = xx.ravel()
X_grid[:, xC_idx] = yy.ravel()

medians = np.median(X, axis=0)
for i in range(X.shape[1]):
    if i not in [WR_idx, xC_idx]:
        X_grid[:, i] = medians[i]

Z = model.predict_proba(X_grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(6, 5))
contourf = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.85)
cbar = plt.colorbar(contourf)
cbar.set_label('Predicted Probability', fontsize=10)

plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1.8, linestyles='-')

scatter = plt.scatter(X_WR, X_xC, c=y, cmap='RdYlBu', edgecolor='white', s=8, alpha=0.65, linewidth=0.2)

legend_elements = scatter.legend_elements()[0]
plt.legend(handles=legend_elements, labels=['Low Risk', 'High Risk'],
           loc='upper right', fontsize=9, frameon=True)

plt.xlabel('$W_R$ (Routine Task Share)', fontsize=11)
plt.ylabel('$x_C$ (Task Clarity)', fontsize=11)
plt.title('Decision Boundary based on $W_R$ and $x_C$', fontsize=12)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.grid(True, alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=300,bbox_inches='tight')
plt.show() 
# 6.1 部分依赖图（Partial Dependence Plot）—— 更专业的决策边界展示
from sklearn.inspection import partial_dependence

# 获取 WR 和 xC 的索引
WR_idx = feature_cols.index('WR') if 'WR' in feature_cols else 0
xC_idx = feature_cols.index('xC') if 'xC' in feature_cols else 3

# 计算部分依赖
pdp = partial_dependence(
    model, X, features=[WR_idx, xC_idx],
    grid_resolution=50,  # 网格密度
    kind='average'       # 平均部分依赖
)

# 提取网格和预测值
xx = pdp['grid_values'][0]   # WR 的网格点
yy = pdp['grid_values'][1]   # xC 的网格点
Z = pdp['average'][0]   # 形状 (len(xx), len(yy))

# 绘图
plt.figure(figsize=(6, 5))

# 填充等高线
contourf = plt.contourf(xx, yy, Z.T, levels=20, cmap='coolwarm', alpha=0.8)
cbar = plt.colorbar(contourf)
cbar.set_label('Partial Dependence\n(Predicted Probability)', fontsize=10)

# 绘制 0.5 等高线（决策边界）
plt.contour(xx, yy, Z.T, levels=[0.5], colors='black', linewidths=1.5)

# 绘制原始样本点（在二维平面上）
X_WR = X[:, WR_idx]
X_xC = X[:, xC_idx]
scatter = plt.scatter(X_WR, X_xC, c=y, cmap='coolwarm', s=6, alpha=0.5, edgecolor='none')
legend_elements = scatter.legend_elements()[0]
plt.legend(handles=legend_elements, labels=['Low Risk', 'High Risk'],
           loc='upper right', fontsize=9)

plt.xlabel('$W_R$ (Routine Task Share)', fontsize=11)
plt.ylabel('$x_C$ (Task Clarity)', fontsize=11)
plt.title('Partial Dependence of $W_R$ and $x_C$ on Predicted Probability', fontsize=10)

plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('partial_dependence.png', dpi=300, bbox_inches='tight')
plt.show()
# 6.2 校准曲线（使用等频分箱，避免样本不均）
plt.figure(figsize=(6, 6))
disp = CalibrationDisplay.from_predictions(y, y_pred_proba_all, n_bins=10, strategy='quantile')
disp.ax_.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
disp.ax_.set_title('Calibration Curve')
disp.ax_.legend(loc='lower right')
disp.ax_.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n可视化结果已保存: prob_distribution.png, decision_boundary.png, calibration_curve.png")

# ==================== 七、模型校准（Platt Scaling） ====================
print("\n" + "=" * 50)
print("模型概率校准（Platt Scaling）")
print("=" * 50)

from sklearn.calibration import CalibratedClassifierCV

# 使用 CalibratedClassifierCV 包装模型
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# 获取校准后的预测概率
y_pred_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

# 评估校准后性能
auc_cal = roc_auc_score(y_test, y_pred_proba_cal)
acc_cal = calibrated_model.score(X_test, y_test)
print(f"校准后 - 准确率: {acc_cal:.4f}, AUC: {auc_cal:.4f}")

# ==================== 八、校准前后对比可视化 ====================
# 8.1 校准曲线对比
from sklearn.calibration import calibration_curve

plt.figure(figsize=(8, 6))

# 校准前
fop_orig, mpv_orig = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='quantile')
plt.plot(mpv_orig, fop_orig, 'o-', label='Before Calibration', linewidth=2, markersize=8)

# 校准后
fop_cal, mpv_cal = calibration_curve(y_test, y_pred_proba_cal, n_bins=10, strategy='quantile')
plt.plot(mpv_cal, fop_cal, 's-', label='After Calibration', linewidth=2, markersize=8)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Calibration Curve Comparison', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=300)
plt.show()

# 8.2 概率分布对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Before Calibration')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_pred_proba_cal, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('After Calibration')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prob_distribution_comparison.png', dpi=300)
plt.show()

print("\n校准结果已保存: calibration_comparison.png, prob_distribution_comparison.png")