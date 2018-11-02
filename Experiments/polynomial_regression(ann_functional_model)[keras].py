# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import os     
os.environ["PATH"] += os.pathsep + 'C:/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin/graphviz'

#################################  准备数据  #################################

X = np.linspace(0,20,2000)
X = X.reshape((2000,1))

# np.random.normal()加入一个正态分布噪音
# https://blog.csdn.net/lanchunhui/article/details/50163669
y = np.power(X,2) + np.random.normal(0,5,(2000,1))  # 二次函数
y = y.reshape((2000,1))

# 标准化
min_max_scaler = MinMaxScaler((0,1))

y = min_max_scaler.fit_transform(y)
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#################################  keras模型  #################################

# This returns a tensor
# 定义一个输入
inputs = Input(shape=(1,), name='input_x')

'''
# 定义（输入）层
# 可使用数据类型、名字等参数
# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')
'''

# 返回值也可以一直为x（当层不需要特别使用时，不需要增加太多变量）
hidden_1 = Dense(10, activation='relu', name='hidden_1')(inputs)
hidden_2 = Dense(6, activation='relu', name='hidden_2')(hidden_1)

# And finally we add the main logistic regression layer
outputs = Dense(1, activation='linear', name='output_y')(hidden_2)

# 定义整个模型
model = Model(inputs=inputs, outputs=outputs)
# 当有多输入/多输出时，可使用列表
# model = Model(inputs=[inputs_a, inputs_b], outputs=[outputs_a, outputs_b])

'''
函数式模型等同于以下序列模型
from keras.models import Sequential

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
# model.add(Activation('relu'))
model.add(Dense(1))
'''

# 编译模型，可以有多个输出，见链接
# https://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
model.compile(optimizer='adam', loss='mse')


###############################  模型训练+预测  ###############################

# 加入数据，训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32)
# 当有多输入/多输出时，可使用列表
# model.fit([X_train], [y_train], epochs=50, batch_size=32)
# 或字典
# model.fit({'input_x': X_train}, {'output_y': y_train}, epochs=50, batch_size=32)


# 用训练好的模型预测
y_pred = model.predict(X_test)

###################################  画图  ###################################

# 测试集和预测结果图
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
ax1.scatter(X_test, y_test, color='r')
ax1.scatter(X_test, y_pred)
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.grid(True)

# 模型结构图
plot_model(model, to_file='keras_functional_model.png',show_shapes=True, show_layer_names=True)