# -*- coding: utf-8 -*-
"""
LASSO路径图 Lasso Path

参考资料
https://blog.csdn.net/Solomon1558/article/details/40951781
https://cosx.org/2011/04/modified-lars-and-lasso

sklearn及画图
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html

http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html
http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html

判断依据
图中从左到右是逐步加入特征的过程中：
先加入，始终有较高coef的特征一般为重要特征
先加入，coef较高又回落的特征是在逐步回归时被剔除的
新引入的自变量和相关系数突然剧烈变化（如出现正负分叉）的自变量可能存在多重共线性
出现反复进出的变量值，说明这些变量存在震荡，应该去除

"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# =============================================================================
# %% 数据导入

from sklearn import datasets

diabetes = datasets.load_diabetes()
X_raw = diabetes.data
y_raw = diabetes.target.reshape(-1,1)

list_X_feature_names = diabetes.feature_names

# =============================================================================

# =============================================================================
# %% 标准化

std_scaler_x = StandardScaler().fit(X_raw)
std_scaler_y = StandardScaler().fit(y_raw)

X_std = std_scaler_x.transform(X_raw)
y_std = std_scaler_y.transform(y_raw).ravel()

# =============================================================================

# =============================================================================
# %% 绘图
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html

'''
# Author: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause
'''

import matplotlib.pyplot as plt

from sklearn import linear_model

X = X_std
y = y_std

print("Computing regularization path using the LARS ...")
_, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

# 在最后一列后标注权重最大的特征
feature_selected_num = 20
coefs_head_index = pd.DataFrame(np.abs(coefs[:,-1])).sort_values(by=0).tail(feature_selected_num).index
list_coefs_feature_names = [list_X_feature_names[i] for i in list(coefs_head_index)]
coefs_head = pd.DataFrame(coefs[:,-1]).iloc[list(coefs_head_index),:]
coefs_head['feature_names'] = list_coefs_feature_names

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.style.use('ggplot')    # 绘图风格
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号

fig = plt.figure(figsize=(16,8), dpi=100)
ax = fig.add_subplot(111)

ax.plot(xx, coefs.T)

# 生成文字标注
for i,f in enumerate(coefs_head[0]):
    ax.text(x=1.01,
            y=f-0.01,
            s=coefs_head.iloc[i,1],
            fontsize=10)
            
ymin, ymax = ax.get_ylim()

# 竖线，方便看特征随路径选出时间
ax.vlines(xx, ymin, ymax, linestyle='dashed', linewidth=0.5)

ax.set_xlabel('|coef| / max|coef|')
ax.set_ylabel('Coefficients')
ax.set_title('LASSO Path')
plt.axis('tight')
ax.axis()

# 图例
# label = list_X_feature_names
# ax.legend(label, loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)

# https://www.cnblogs.com/nju2014/p/5707980.html
plt.subplots_adjust(left=0.05, right=0.95)

plt.show()
# =============================================================================
