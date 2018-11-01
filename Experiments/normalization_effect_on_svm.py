# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVR

boston = load_boston()

X_raw = boston.data
y_raw = boston.target

scaler_x = MinMaxScaler(feature_range=(0,1)).fit(X_raw)
scaler_y = MinMaxScaler(feature_range=(0,1)).fit(y_raw.reshape(-1,1))

X_std = scaler_x.transform(X_raw)
y_std = scaler_y.transform(y_raw.reshape(-1,1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.33, random_state=32)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y_std, test_size=0.33, random_state=32)


###################### 验证estimator的score是否单纯与归一化相关 ######################

svr = LinearSVR(C=0.5)

# STEP1 计算归一化前后模型的r2得分
svr.fit(X_train, y_train)
svr_y_pred = svr.predict(X_test)

# 计算r2/均方差
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

svr_r2 = r2_score(y_test, svr_y_pred)
svr_score = svr.score(X_test, y_test)
print('svr_r2: {}\n'.format(svr_r2))
print('svr_score: {}\n'.format(svr_score))
print('结果：svr_r2 == svr_v_score, 两者为同一种指标\n')


svr_std_x = LinearSVR(C=0.5)
svr_std_x.fit(X_train_std, y_train)
svr_std_x_y_pred = svr_std_x.predict(X_test_std)

svr_std_x_score = svr_std_x.score(X_test_std, y_test)
svr_std_x_mse = mean_squared_error(y_test, svr_std_x_y_pred)

print('svr_std_x_score: {}\n'.format(svr_std_x_score))
print('svr_std_x_mse: {}\n'.format(svr_std_x_mse))


svr_std = LinearSVR(C=0.5)
svr_std.fit(X_train_std, y_train_std)
svr_std_y_pred = svr_std.predict(X_test_std)

svr_std_score = svr_std.score(X_test_std, y_test_std)
svr_std_r2 = r2_score(y_test_std, svr_std_y_pred)
svr_std_mse = mean_squared_error(y_test_std, svr_std_y_pred)

print('svr_std_r2: {}\n'.format(svr_std_r2))
print('svr_std_score: {}\n'.format(svr_std_score))
print('svr_std_mse: {}\n'.format(svr_std_mse))

svr_inv_y_pred = scaler_y.inverse_transform(svr_std_y_pred.reshape(-1,1))
svr_inv_r2 = r2_score(y_test, svr_inv_y_pred)
svr_inv_mse = mean_squared_error(y_test, svr_inv_y_pred)

print('svr_inv_r2: {}\n'.format(svr_inv_r2))
print('svr_inv_mse: {}\n'.format(svr_inv_mse))
print('将数据归一化后反归一化，相关系数r2不变，说明r2的提升与归一化后的模型计算相关、与数据本身变换无关\n'\
      '数据均方差有变化，说明模型自带的得分函数score计算【反归一化】之后的实际r2系数')