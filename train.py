import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,KernelPCA,IncrementalPCA,SparsePCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from scipy.stats import mode
# 加载数据
data = pd.read_csv('data.csv')

# 数据预处理
# 去除ID列
data.drop(columns=['ID'], inplace=True)

# 分离特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维
# pca = PCA(n_components=0.95)  # 保留95%的方差
# pca = IncrementalPCA(n_components=100)
# pca = KernelPCA(n_components=100, kernel='rbf', gamma=15, random_state=42)
# pca = SparsePCA(n_components=100, random_state=42)
# tsne = TSNE(n_components=100, random_state=42)
# X_pca = pca.fit_transform(X_scaled)
y = np.where(y == 'H', 0, 1)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
dimensionality_reduction_methods = {
    'PCA': PCA(n_components=0.95, random_state=42),
    'IncrementalPCA': IncrementalPCA(n_components=100),
    'KernelPCA': KernelPCA(n_components=100, kernel='rbf', gamma=15, random_state=42),
    'SparsePCA': SparsePCA(n_components=100, random_state=42),
    'TSNE': TSNE(n_components=2, random_state=42),  # TSNE通常用于2D或3D可视化
    'LDA': LinearDiscriminantAnalysis()
}
# 定义分类器列表和参数字典
classifiers = {
    "XGBoost": (xgb.XGBClassifier(), {}),
    "AdaBoost": (AdaBoostClassifier(), {'n_estimators': 200}),
    "Gradient Boosting Machine": (GradientBoostingClassifier(), {}),
    "LightGBM": (lgb.LGBMClassifier(), {}),
    "SVM": (SVC(), {'C': 10.0, 'gamma': 'scale'}),
    "Random Forest": (RandomForestClassifier(), {'n_estimators': 200, 'max_depth': None, 'max_features': 'sqrt'}),
    "MLP": (MLPClassifier(), {'hidden_layer_sizes': (500, 500), 'max_iter': 500}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': 3}),
    "Decision Tree": (DecisionTreeClassifier(), {'max_depth': None, 'min_samples_split': 2}),
    "Logistic Regression": (LogisticRegression(), {'C': 10.0}),
    "Naive Bayes": (GaussianNB(), {'var_smoothing': 1e-9}),
    "LDA": (LinearDiscriminantAnalysis(), {}),
}

# 训练分类器并评估准确率
accuracies = {}
for name, (clf, params) in classifiers.items():
    clf.set_params(**params)  # 设置分类器参数
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# 特别处理 K-means
kmeans = KMeans(n_clusters=len(np.unique(y_train)), random_state=42)
kmeans.fit(X_train)
# 将聚类中心分配给最频繁的类标签
clusters_labels = [np.bincount(y_train[kmeans.labels_ == i]).argmax() for i in range(kmeans.n_clusters)]
y_pred_kmeans = [clusters_labels[cluster] for cluster in kmeans.predict(X_test)]
accuracies["K-means"] = accuracy_score(y_test, y_pred_kmeans)

# 输出每个分类器的准确率
for name, accuracy in accuracies.items():
    print(f"{name}: {accuracy:.4f}")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
X_pca = tsne.fit_transform(X_scaled)

# 绘制数据可视化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("t-SNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()


