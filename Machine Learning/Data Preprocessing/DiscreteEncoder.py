# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

df_data = pd.DataFrame({'name':['Chris','John','David','Tom'],
                        'age':[18,23,35,52],
                        'height':[176,188,168,182],
                        'job':['student','student','teacher','worker']})


# get_dummies() 
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
dummy_result = pd.get_dummies(df_data)
print('get_dummies() Result:\n', dummy_result, '\n')
dummy_result_column = pd.get_dummies(df_data['age'], prefix='age')
print('get_dummies(column) Result:\n', dummy_result_column, '\n')


# One-Hot-Encoder 
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
# OneHotEncoder只支持确定尺寸的数据，故若现有数据为Dataframe/Series，需要：
# np.array(Dataframe/Series).reshape()或 Dataframe/Series.values.reshape()
oh_result = oh.fit_transform(np.array(df_data['age']).reshape(-1,1))
oh_result = oh_result.toarray()
print('One-Hot-Encoder Result:\n', oh_result, '\n')


# LabelBinarizer
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb_result = lb.fit_transform(df_data['name'])
print('LabelBinarizer Result:\n', lb_result, '\n')


# LabelEncoder
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_result = le.fit_transform(df_data['job'])
df_data_le = df_data.copy()
df_data_le['job num'] = le_result
print('LabelEncoder Result:\n', df_data_le, '\n')


'''
暂无适用场景

# DictVectorizer
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

# MultiLabelBinarizer
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
'''