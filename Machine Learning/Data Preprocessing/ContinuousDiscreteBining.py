# -*- coding: utf-8 -*-


# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
# wine数据集
from sklearn.datasets import load_wine
wine = load_wine()

X = wine.data
y = wine.target

import pandas as pd

df_X = pd.DataFrame(X).iloc[0:10,4]
df_X.describe()

# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
# cut分割数值，保留数值区间
x_cut_simple = pd.cut(df_X,3)

# cut分割数值区间，赋值整数标签
x_cut_intlabel = pd.cut(df_X,3, labels=False)

# 生成哑变量
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
x_dummy = pd.get_dummies(x_cut_simple, prefix='cut')

# cut分割数值区间，赋值自定义标签
x_cut_label = pd.cut(df_X,3, labels=['S','M','L'])

# cut分割数值区间，增加返回array
x_cut,arr_cut = pd.cut(df_X,3, retbins=True)

# cut分割数值阈值
x_cut_numeric = pd.cut(df_X, [90, 100, 110])

# cut分割数值阈值，minmax，调整小数点
x_cut_minmax = pd.cut(df_X, [df_X.min()*0.99,110,df_X.max()*1.01], precision=0)


# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
# qcut分割样本数量
x_qcut = pd.qcut(df_X,5)

# qcut分割样本分位数
x_qcut = pd.qcut(df_X,[0, .2, .4, .6, .8, 1.])