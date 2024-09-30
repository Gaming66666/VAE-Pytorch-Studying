import xgboost as xgb
#XGBoost
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
# 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=82)

# 将数据转换为 DMatrix 对象，这是 XGBoost 专用的数据结构
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
param = {
    'max_depth': 12,  # 树的最大深度
    'eta': 0.1,  # 学习率
    'objective': 'multi:softmax',  # 多分类问题
    'num_class': 4  # 类别数
}
num_round = 20  # 训练轮数

# 训练模型
bst = xgb.train(param, dtrain, num_round)

# 预测
preds = bst.predict(dtest)

# 计算准确率
accuracy = accuracy_score(y_test, preds)
print(f'Accuracy: {accuracy:.2f}')

# 可视化
# 为了可视化，我们只取前两个特征
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# 训练一个新的模型，只使用前两个特征
dtrain_2d = xgb.DMatrix(X_train_2d, label=y_train)
dtest_2d = xgb.DMatrix(X_test_2d, label=y_test)
bst_2d = xgb.train(param, dtrain_2d, num_round)

# 预测
preds_2d = bst_2d.predict(dtest_2d)


# 绘制决策边界
def plot_decision_boundaries(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(xgb.DMatrix(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)
    plt.show()


plot_decision_boundaries(X_test_2d, y_test, bst_2d, 'XGBoost Decision Boundaries')