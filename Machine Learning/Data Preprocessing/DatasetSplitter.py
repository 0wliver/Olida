# -*- coding: utf-8 -*-


# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
# wine数据集
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


# 原始K-fold 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
# 输入一个样本数为n的数据（使用X即可，因X,y样本数相同），返回分割后的 索引向量（生成器形式，需要用for依次获得每次分割结果）
# 参数：shuffle打乱； random_state随机种子； kf.get_n_splits(X)获得折数
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=33)
kf_count = 0
for train_index, test_index in kf.split(X):
   X_train_kf, X_test_kf = X[train_index], X[test_index]
   y_train_kf, y_test_kf = y[train_index], y[test_index]
   print("KFold Num:", kf_count, 
         "TRAIN:", train_index.shape[0], 
         "TEST:", test_index.shape[0])
   kf_count += 1
   # 此处开始处理数据，并记录每次结果后，求最终平均


# 分层K-fold 
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
# 分层指：按照（分类）结果将不同取值按百分比放入各折中，防止某折样本中缺少某个分类导致结果很差
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
skf_count = 0
for train_index, test_index in skf.split(X,y):  # 此处有y
   X_train_skf, X_test_skf = X[train_index], X[test_index]
   y_train_skf, y_test_skf = y[train_index], y[test_index]
   print("StratifiedKFold Num:", skf_count, 
         "TRAIN:", train_index.shape[0], 
         "TEST:", test_index.shape[0])
   skf_count += 1
   # 此处开始处理数据，并记录每次结果后，求最终平均