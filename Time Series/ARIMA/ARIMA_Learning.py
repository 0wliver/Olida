
# coding: utf-8

# # ARIMA 探究 #

# ## a.    数据预处理 ##
# ### 导入包 ###

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
import numpy as np


# ### 格式转换 ###
# 将数据格式转换为 索引为Datetime的Pandas.Series，否则无法进行ADF校验

# In[2]:


origin_data = pd.read_csv('AirPassengers.csv')

data = origin_data.set_index(keys='Month', drop=True)
data.rename(columns={'#Passengers':'Passengers'}, inplace=True)

data.index = pd.to_datetime(data.index)
data = data.Passengers


# ### 对数化 ###
# 由于原数据值域范围比较大，为了缩小值域，同时保留其他信息，常用的方法是对数化，取log

# In[3]:


data_log = np.log(data)

# fitted_data = np.exp(fitted_data_log)    # 逆向操作


# ## b.    平稳性验证（ADF校验） ##
# 对整体数据进行ADF校验

# In[4]:


import statsmodels.tsa.stattools
adf_result = statsmodels.tsa.stattools.adfuller(data_log)    # 对乘客数做ADF校验

#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in adf_result[4].items():
    adf_output['Critical Value (%s)'%key] = value
print(adf_output)


# ### 使用seansonal包之前，要先进行index的格式转换，通过to_datetime(df.index)将索引转换为datetime ###

# In[5]:


# http://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html
import statsmodels.tsa.seasonal
decomposition = statsmodels.tsa.seasonal.seasonal_decompose(data_log, model='additive')    
# 模型分为additive（相加）, multiplicative（相乘），可以都试一下看效果
# additive: The additive model is Y[t] = T[t] + S[t] + e[t]
# multiplicative: The multiplicative model is Y[t] = T[t] * S[t] * e[t]

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.plot(data_log, label='Original')
ax1.legend(loc='best')

ax2.plot(trend, label='Trend')
ax2.legend(loc='best')

ax3.plot(seasonal,label='Seasonality')
ax3.legend(loc='best')

ax4.plot(residual, label='Residuals')
ax4.legend(loc='best')

plt.tight_layout()    # 调节绘图空间使图像比较自然


# ### 滑动平均校验稳定性 ###

# In[6]:


# 注：pandas新版本中将rolling_mean等方法统一为df.rolling()方法，并调用mean()求滑动的平均值进行替代
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html
data_log_rolling_mean = data_log.rolling(window=12).mean()
data_log_rolling_mean_diff = data_log - data_log_rolling_mean
data_log_rolling_mean_diff.dropna(inplace = True)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

ax.plot(data_log_rolling_mean_diff)
ax.plot(data_log_rolling_mean_diff.rolling(window=12).mean())
ax.plot(data_log_rolling_mean_diff.rolling(window=12).std())

ax.legend(['data_log_rolling_diff', 'data_log_rolling_diff\'s mean', 'data_log_rolling_diff\'s std'])


# ### 指数平均校验稳定性 ###

# In[7]:


# 注：pandas新版本中将ewma(Exponentially Weighted Moving-Average 指数平均)等方法统一为df.ewm()方法，并调用mean()求指数滑动的平均值进行替代
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ewm.html
data_log_ewm_mean = data_log.ewm(halflife=12).mean()
data_log_ewm_mean_diff = data_log - data_log_ewm_mean
data_log_ewm_mean_diff.dropna(inplace = True)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

ax.plot(data_log_ewm_mean_diff)
ax.plot(data_log_ewm_mean_diff.rolling(window=12).mean())
ax.plot(data_log_ewm_mean_diff.rolling(window=12).std())

ax.legend(['data_log_ewm_diff', 'data_log_ewm_diff\'s mean', 'data_log_ewm_diff\'s std'])


# ### 差分校验稳定性 ###

# In[8]:


# 注：pandas新版本中使用df.diff求差分（参数periods为阶数），替代data - data.shift()操作
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.diff.html
data_log_minus_diff =  data_log.diff(periods=1)
data_log_minus_diff.dropna(inplace=True)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)

ax.plot(data_log_minus_diff)
ax.plot(data_log_minus_diff.rolling(window=12).mean())
ax.plot(data_log_minus_diff.rolling(window=12).std())

ax.legend(['data_log_minus_diff', 'data_log_minus_diff\'s mean', 'data_log_minus_diff\'s std'])


# ### 对一阶差值做ADF校验 ###

# In[9]:


import statsmodels.tsa.stattools
adf_result = statsmodels.tsa.stattools.adfuller(data_log.diff(1).dropna())    # 对一阶差值做ADF校验

#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test of DIFF(2):')
adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in adf_result[4].items():
    adf_output['Critical Value (%s)'%key] = value
print(adf_output)


# ## c.    非白噪声验证（Ljung-Box检验） ##
# Ljung-Box检验：假设检验：原假设目标数列为高斯白噪声，当p值>5%或10%认为接受该检验，否则拒绝该检验，目标数列存在时序相关关系

# In[10]:


import statsmodels.stats.diagnostic
lb, pvalue = statsmodels.stats.diagnostic.acorr_ljungbox(data_log.diff(1).dropna(), lags=24)
print(lb,pvalue)


# ## d. 画ACF, PACF图确定参数 ##
# 注：二阶差分不是diff(2)而是diff(1).diff(1)！，diff(2)是Yt3 - Yt1

# In[11]:


import statsmodels.graphics.tsaplots
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = statsmodels.graphics.tsaplots.plot_acf(data_log.diff(1).diff(1).dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = statsmodels.graphics.tsaplots.plot_pacf(data_log.diff(1).diff(1).dropna(),lags=40,ax=ax2)


# ### 还可采用建立多模型，通过比较AIC BIC HIC，全部最小的为最佳模型的方法 ###
# https://blog.csdn.net/u010414589/article/details/49622625  <br />
# https://blog.csdn.net/hal_sakai/article/details/51965657

# In[12]:


'''
arma_mod20 = sm.tsa.ARMA(dta,(7,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
'''


# ## e. 建模并拟合 ##

# In[13]:


import statsmodels.tsa.arima_model

# https://blog.csdn.net/wangqi_qiangku/article/details/79384731
arima_model = statsmodels.tsa.arima_model.ARMA(data_log.diff(1).dropna(),(11,2), freq='MS').fit()

# 二阶
# arima_model = statsmodels.tsa.arima_model.ARMA(data_log.diff(1).diff(1).dropna(),(8,1), freq='MS').fit()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)

ax.plot(data_log.diff(1).diff(1))
ax.plot(arima_model.fittedvalues, color='red')

rss = sum((arima_model.fittedvalues-data_log.diff(1).dropna()).dropna()**2)

# 二阶
# rss = sum((arima_model.fittedvalues-data_log.diff(1).diff(1).dropna()).dropna()**2)

ax.set_title('RSS: %.4f'% rss)


# ## f.    模型检验 ##
# 主要针对模型拟合后的残差 arima_model.resid做检验，满足ACF+PACF、Ljung-Box、独立性、QQ图等

# In[14]:


resid = arima_model.resid

'''
注：
拟合出arma模型的参数：arima_model.arparams(ar), arima_model.maparams(ma)，给定阶数arima_model.k_ar, arima_model.k_ma
model.predict() 返回的值就是 fittedvalues
'''


# ### ACF, PACF图 ###
# 观察连续残差是否（自）相关

# In[15]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = statsmodels.graphics.tsaplots.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = statsmodels.graphics.tsaplots.plot_pacf(resid, lags=40, ax=ax2)


# ### D-W检验 ###
# 德宾-沃森（Durbin-Watson）检验，简称D-W检验，是目前检验自相关性最常用的方法，但它只使用于检验一阶自相关性。因为自相关系数ρ的值介于-1和1之间，所以 0 ≤ DW ≤ ４。并且DW＝0＝＞ρ＝１ 即存在正自相关性 
# DW＝４＜＝＞ρ＝－１　即存在负自相关性 
# DW＝２＜＝＞ρ＝0　　即不存在（一阶）自相关性 
# 因此，当DW值显著的接近于0或４时，则存在自相关性，而接近于２时，则不存在（一阶）自相关性。这样只要知道DW统计量的概率分布，在给定的显著水平下，根据临界值的位置就可以对原假设H0进行检验

# In[16]:


import statsmodels.stats.stattools
print(statsmodels.stats.stattools.durbin_watson(resid.values))


# ### QQ图 ###
# 用于直观验证一组数据是否来自某个分布（此处使用正态分布），或者验证某两组数据是否来自同一（族）分布

# In[17]:


import statsmodels.graphics.gofplots
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = statsmodels.graphics.gofplots.qqplot(resid, line='q', ax=ax, fit=True)


# ### Ljung-Box检验 ###
# Ljung-Box test是对randomness的检验,或者说是对时间序列是否存在滞后相关的一种统计检验。
# 对于滞后相关的检验，我们常常采用的方法还包括计算ACF和PCAF并观察其图像，但是无论是ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在。 
# 时间序列中一个最基本的模型就是高斯白噪声序列。而对于ARIMA模型，其残差被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，拟合后我们要对残差的估计序列进行LB检验，判断其是否是高斯白噪声，如果不是，那么就说明ARIMA模型也许并不是一个适合样本的模型。

# In[18]:


import statsmodels.stats.diagnostic
lb, pvalue = statsmodels.stats.diagnostic.acorr_ljungbox(resid, lags=40)
test_data = np.c_[range(1,41), lb, pvalue]
table = pd.DataFrame(test_data, columns=['lag', "Q", "Prob(>Q)"])
print(table.set_index('lag'))

'''
两段代码作用相同
r,q,p = statsmodels.tsa.stattools.acf(resid.values.squeeze(), qstat=True)
test_data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(test_data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
'''


# ## g.    模型预测 ##

# In[19]:


# 一阶还原
def reverse_order_1(arima_model, data_log):
    fitted_values = arima_model.fittedvalues
    data_log_origin = data_log
    data_log_diff_1_first = pd.Series(data=data_log_origin[0], index=data_log_origin.index)
    data_log_diff_1_fitted = data_log_diff_1_first.add(fitted_values.cumsum(), fill_value=0)
    data_fitted = np.exp(data_log_diff_1_fitted)
    return data_fitted


# 一阶预测 + 还原
def reverse_order_1_predict(arima_model, start_date, end_date, data_log):
    fitted_values = arima_model.predict(start_date, end_date)
    data_log_origin = data_log
    
    # 生成start_date - end_date的索引，MS=month start
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html
    data_log_diff_1_first = pd.Series(data=data_log_origin[start_date], index=pd.date_range(start=start_date, end=end_date, freq='MS'))
    
    data_log_diff_1_fitted = data_log_diff_1_first.add(fitted_values.cumsum(), fill_value=0)
    data_fitted = np.exp(data_log_diff_1_fitted)
    return data_fitted


# 二阶还原
def reverse_order_2(arima_model, data_log):
    fitted_values = arima_model.fittedvalues
    data_log_origin = data_log
    data_log_diff_2_first = pd.Series(data=data_log_origin.diff(1).dropna()[0], index=data_log_origin.diff(1).dropna().index)
    data_log_diff_2_fitted = data_log_diff_2_first.add(fitted_values.cumsum(), fill_value=0)
    
    # 与原数据比较，验证还原准确性
    # data_log_diff_2_fitted_compare = data_log_diff_2_first.add(data_log.diff(1).diff(1).cumsum(), fill_value=0)%%!

    data_log_diff_1_first = pd.Series(data=data_log_origin[0], index=data_log_origin.index)
    data_log_diff_1_fitted = data_log_diff_1_first.add(data_log_diff_2_fitted.cumsum(), fill_value=0)
    
    # 与原数据比较，验证还原准确性
    # data_log_diff_1_fitted_compare = data_log_diff_1_first.add(data_log_diff_2_fitted_compare.cumsum(), fill_value=0)
    
    data_fitted = np.exp(data_log_diff_1_fitted)
    return data_fitted

reversed_fittedvalue = reverse_order_1(arima_model, data_log)
reversed_predictedvalue = reverse_order_1_predict(arima_model, '1949-02-01', '1964-09-01', data_log)

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax.plot(data)
ax.plot(reversed_fittedvalue)
ax.plot(reversed_predictedvalue)
ax.legend(['origin_data', 'fitted_data', 'predicted_data'])

# rss = np.sqrt(sum((data_fitted-data)**2)/len(data))
# ax.set_title('RSS: %.4f' % rss)

