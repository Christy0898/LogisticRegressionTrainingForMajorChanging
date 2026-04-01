# LogisticRegressionTrainingForMajorChanging
个人第一篇机器学习论文源代码

# 首先生成数据。
在python中运行generate_data.py，即生成20000个数据。
如果需要修改数据的生成逻辑，在代码中直接修改即可。

# 其次运行logistic_regression_training_optimization.py
程序会依次进行数据集划分、梯度下降和参数更新（打印损失函数）、全特征训练模型、模型评估、保存模型参数和可视化成果（绘制决策边界图、预测概率分布图、校准曲线）；
然后，程序会进行下一步的概率校准（Platt Scaling），并经习惯校准前后效果的可视化，生成新的决策边界图、校准曲线对比图、预测概率对比图。

# 输出的图像均已经上传
