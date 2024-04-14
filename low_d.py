from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
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
# 假设 X_scaled 和 y 已经准备好，并且 y 已经是二进制标签
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
# 定义降维方法列表
dimensionality_reduction_methods = {
    'PCA': PCA(n_components=0.95, random_state=42),
    'IncrementalPCA': IncrementalPCA(n_components=100),
    'KernelPCA': KernelPCA(n_components=100, kernel='rbf', gamma=15, random_state=42),
    'SparsePCA': SparsePCA(n_components=100, random_state=42),
    # 'TSNE': TSNE(n_components=10, random_state=42),  # TSNE通常用于2D或3D可视化
    'LDA': LinearDiscriminantAnalysis(),
    'MDS': MDS(n_components=2, random_state=42)
}
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
y = np.where(y == 'H', 0, 1)
# 初始化准确率记录字典
# 初始化准确率记录字典
accuracies = {name: [] for name in classifiers.keys()}

# 对每种降维方法进行迭代
for dr_name, dr_method in dimensionality_reduction_methods.items():
    # 应用降维方法
    if dr_name == 'LDA':
        # LDA 需要目标变量 y
        X_reduced = dr_method.fit_transform(X_scaled, y)
    else:
        X_reduced = dr_method.fit_transform(X_scaled)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    # 对每个分类器进行迭代
    for clf_name, (clf, params) in classifiers.items():
        # 训练分类器
        clf.set_params(**params)
        clf.fit(X_train, y_train)

        # 预测测试集
        y_pred = clf.predict(X_test)

        # 计算准确率
        acc = accuracy_score(y_test, y_pred)

        # 记录准确率
        accuracies[clf_name].append(acc)

    # 输出每种降维方法下的分类器准确率
    print(f"Accuracies for {dr_name}:")
    for clf_name in classifiers.keys():
        print(f"{clf_name}: {accuracies[clf_name][-1]:.4f}")

# 计算每种分类器的平均准确率
average_accuracies = {clf_name: np.mean(accs) for clf_name, accs in accuracies.items()}
print("\nAverage accuracies across different dimensionality reduction methods:")
for clf_name, avg_acc in average_accuracies.items():
    print(f"{clf_name}: {avg_acc:.4f}")