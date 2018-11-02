# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

from xgboost.sklearn import XGBRegressor

# =============================================================================
# 导入数据

boston = load_boston()
data_raw = boston.data
target_raw = boston.target

X_train, X_test, y_train, y_test = train_test_split(data_raw, target_raw, test_size=0.1, random_state=33)
feature_names = boston.feature_names

# =============================================================================

# =============================================================================
# 建模并训练

# 封装为sklearn格式的模型
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
xgb = XGBRegressor()

xgb.fit(X_train, y_train, 
        early_stopping_rounds=10, # 当n次结果没有变小时提前终止
        eval_set=[(X_train, y_train), (X_test, y_test)], # 评价集，类似于predict
        eval_metric=['rmse']
        # 评价指标 'error','logloss','rmse','auc'等
        # http://xgboost.readthedocs.io/en/latest/parameter.html 
        )

# =============================================================================

# =============================================================================
# 预测并评价

y_pred = xgb.predict(X_test) # 预测用于实际模型，以及进行更多的分析，如auc

# score_accuracy = accuracy_score(y_test, y_pred) # accuracy只用于分类
score_r2 = r2_score(y_test, y_pred)
score_mse = mean_squared_error(y_test, y_pred)

# retrieve performance metrics
results = xgb.evals_result_

feature_importance = xgb.feature_importances_

# =============================================================================

# =============================================================================
# 画图

# 误差下降图
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE on Dataset-Boston')
plt.show()

'''
# 参数重要性图

fig, ax = plt.subplots()
ax.bar(range(len(feature_importance)), feature_importance, label='Feature Importance')
ax.legend()
plt.xticks(range(len(feature_importance)), feature_names)
plt.ylabel('RMSE')
plt.title('XGBoost RMSE on Dataset-Boston')
plt.show()
'''
# =============================================================================



