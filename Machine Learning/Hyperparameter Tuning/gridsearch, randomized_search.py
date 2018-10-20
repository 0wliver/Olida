# -*- coding: utf-8 -*-

import scipy

from sklearn.datasets import load_boston
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import LinearSVR

boston = load_boston()

X_raw = boston.data
y_raw = boston.target

scaler_x = MinMaxScaler(feature_range=(0,1)).fit(X_raw)
scaler_y = MinMaxScaler(feature_range=(0,1)).fit(y_raw.reshape(-1,1))

X_std = scaler_x.transform(X_raw)
y_std = scaler_y.transform(y_raw.reshape(-1,1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.33, random_state=33)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y_std, test_size=0.33, random_state=33)

param_grid = {
        # 'C': random.uniform(0,1),
        'C': [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1]
        }

# scipy.stats.uniform(0,1) 返回0-1中的float，
# scipy.stats.randint(0,100) 返回0-100中的int
param_distributions = {
        'C': scipy.stats.uniform(0,1),
        # 'max_iter' : scipy.stats.randint(1000,2000)
        }

################################ GridSearchCV ################################ 

svr = LinearSVR()

regr_grid = GridSearchCV(svr, param_grid=param_grid, cv=5)
regr_grid.fit(X_train_std, y_train_std)

print('Best Score(Grid): {:.4f}'.format(regr_grid.best_score_))
print('Best Params(Grid): {}\n'.format(regr_grid.best_params_))

################################ RandomizedSearchCV ################################ 

svr = LinearSVR()

regr_rand = RandomizedSearchCV(svr, param_distributions=param_distributions, n_iter=7)
regr_rand.fit(X_train_std, y_train_std)

print('Best Score(Rand): {:.4f}'.format(regr_rand.best_score_))
print('Best Params(Rand): {}\n'.format(regr_rand.best_params_))

