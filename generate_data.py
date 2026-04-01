import numpy as np
import pandas as pd
from scipy.stats import dirichlet

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 参数设置
n_samples = 20000
noise_std = 0.03  # 噪声水平

# 定义“真实”系数（符合理论预期：WR, xC, 以及交互项为正）
beta_true = np.array([
    2.0,   # WR
    0.0,   # WM (理论上影响不显著，设为0)
   -1.6,   # WV
    2.4,   # xC
    0.0,   # xE
   -1.0,   # xS
   -1.2,   # xH
    3.0,   # xC*WR
   -1.6,   # xH*WV
    1.2,   # WR*WV
   -0.6,   # WM*WV
   -1.0,   # xC*xS
    0.8    # xE*xH
])
beta0 = -0.5  # 截距

# 生成特征
def generate_features(n):
    # 任务占比：Dirichlet分布，和为1
    task_ratios = dirichlet.rvs(alpha=[2,2,2], size=n)  # (n,3)
    WR, WM, WV = task_ratios[:,0], task_ratios[:,1], task_ratios[:,2]
    
    # 评分特征：均匀分布
    xC = np.random.uniform(0, 1, n)
    xE = np.random.uniform(0, 1, n)
    xS = np.random.uniform(0, 1, n)
    xH = np.random.uniform(0, 1, n)
    
    # 交互项
    X = np.column_stack([
        WR, WM, WV,
        xC, xE, xS, xH,
        xC * WR,
        xH * WV,
        WR * WV,
        WM * WV,
        xC * xS,
        xE * xH
    ])
    return X, WR, WM, WV, xC, xE, xS, xH

X, WR, WM, WV, xC, xE, xS, xH = generate_features(n_samples)

# 计算真实概率
z = beta0 + X @ beta_true
p_true = 1 / (1 + np.exp(-z))

# 加入高斯噪声
p_noisy = p_true + np.random.normal(0, noise_std, n_samples)
p_noisy = np.clip(p_noisy, 0, 1)  # 确保在[0,1]内

# 生成二分类标签
Y = np.random.binomial(1, p_noisy)

# 构建DataFrame并保存
df = pd.DataFrame(X, columns=[
    'WR', 'WM', 'WV', 'xC', 'xE', 'xS', 'xH',
    'xC_WR', 'xH_WV', 'WR_WV', 'WM_WV', 'xC_xS', 'xE_xH'
])
df['Y'] = Y
df.to_csv('data_for_logistic_regression.csv', index=False)

print(f"数据已生成，样本量 {len(df)}，高风险比例 {Y.mean():.2%}")

# ==================== 3σ 异常值剔除 ====================
# 对数值特征列进行异常值检测（排除标签列）
feature_cols = ['WR', 'WM', 'WV', 'xC', 'xE', 'xS', 'xH']
initial_n = len(df)

# 初始化剔除掩码
mask = pd.Series([True] * initial_n)

for col in feature_cols:
    mean = df[col].mean()
    std = df[col].std()
    col_mask = (df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)
    mask = mask & col_mask

# 应用掩码
df_clean = df[mask].reset_index(drop=True)

print(f"原始样本量：{initial_n}")
print(f"剔除异常值后样本量：{len(df_clean)}")
print(f"剔除比例：{(initial_n - len(df_clean))/initial_n:.2%}")

# 保存清洗后的数据
df_clean.to_csv('data_for_logistic_regression.csv', index=False)