# -*- coding: utf-8 -*-


# wine数据集
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
from sklearn.datasets import load_wine
wine = load_wine()

X = wine.data
y = wine.target


# train_test_split
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=33)
print("Train_Test_Split", 
      "TRAIN:", X_train_sp.shape[0], 
      "TEST:", X_test_sp.shape[0])


# 标准化 
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler
standard_scaler=StandardScaler(copy=True, with_mean=True, with_std=True)
x_0_s = standard_scaler.fit_transform(X_train_sp[:,0].reshape(-1,1))
print("StandardScaler ", "Value Mean:", standard_scaler.mean_)
print("StandardScaler ", "Value Var:", standard_scaler.var_)

x_0_s_reverse = standard_scaler.inverse_transform(x_0_s)    # 还原操作


# 归一化 
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler(copy=True, feature_range=(x_0_s.min(), x_0_s.max()))
# minmax_scaler = MinMaxScaler(copy=True, feature_range=(0,1)) # 实际处理时常使用0,1

# 选择数据第0列，reshape(-1,1)将其重塑为列向量，归一化并应用
# 也可以使用scaler.fit, scaler.transform分布完成 归一化 + 应用的操作
x_0_m = minmax_scaler.fit_transform(X_train_sp[:,0].reshape(-1,1))
print("MinMaxScaler ", "Value Max:", minmax_scaler.data_max_) # 最大值
print("MinMaxScaler ", "Value Min:", minmax_scaler.data_min_) # 最小值
print("MinMaxScaler ", "Value Range:", minmax_scaler.data_range_) # 极差

x_0_m_reverse = minmax_scaler.inverse_transform(x_0_m)    # 还原操作


################################## 画图 ##################################
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(131)
ax1.set_title("origin distribution")
#ax1.set_xlim(10,15)
ax2 = fig.add_subplot(132)
ax2.set_title("minmax scaler")
ax2.set_xlim(-3,3)
ax3 = fig.add_subplot(133)
ax3.set_title("standard scaler")
ax3.set_xlim(-3,3)

ax1.scatter(X_train_sp[:,0].reshape(-1,1), X_train_sp[:,1].reshape(-1,1))
ax2.scatter(x_0_m, X_train_sp[:,1].reshape(-1,1))
ax3.scatter(x_0_s, X_train_sp[:,1].reshape(-1,1))

plt.show()


