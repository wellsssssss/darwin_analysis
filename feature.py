import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 假设我们有一个数据集和对应的特征名称
data = pd.read_csv('data.csv')
data.drop(columns=['ID'], inplace=True)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 如果标签是非数字，将其转换为数字
y = np.where(y == 'H', 0, 1)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 训练随机森林模型
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_scaled, y)

# 获取特征重要性
importances = rf.feature_importances_

# 对特征重要性进行排序
indices = np.argsort(importances)[::-1]

# 选择最重要的前20个特征
top_n = 20
top_indices = indices[:top_n]

# 指定的x轴标签
x_labels = [
    "total time 23", "total time 15", "air time 15", "air time 23",
    "paper time 23", "total time 3", "total time 6", "air time 17",
    "total time 17", "total time 7"
]

# 确保提供的x轴标签在数据集的特征中
# assert all(label in X.columns for label in x_labels), "Some labels are not in the dataset"

# 绘制特征重要性的柱状图，只包括最重要的前20个特征
plt.figure(figsize=(12, 6))
plt.title("Top 20 Feature Importances by RandomForestClassifier")
bars = plt.bar(range(len(x_labels)), importances[top_indices[:len(x_labels)]], align="center")

# 设置柱状图的颜色
for i, bar in enumerate(bars):
    bar.set_color('C0' if i % 2 == 0 else 'C1')  # C0为淡蓝色，C1为橙色

plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
plt.xlim([-1, len(x_labels)])
plt.tight_layout()
plt.show()

# 打印每个特征及其对应的贡献值
print("Top Feature Importances:")
for i, label in zip(top_indices[:len(x_labels)], x_labels):
    print(f"{label}: {importances[i]:.4f}")