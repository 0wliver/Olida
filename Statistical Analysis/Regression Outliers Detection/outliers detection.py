
# coding: utf-8

# # 回归异常值诊断 Regression Outliers Detection #

# ## 名词说明 ##
# https://github.com/0wliver/Olida/tree/master/Statistical%20Analysis/Regression%20Outliers%20Detection

# In[1]:


import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd


# ## 样本 ##
# **犯罪率和其它变量的关系数据集**
# 
# https://stats.idre.ucla.edu/r/dae/robust-regression/

# In[2]:


origin_data = pd.read_stata(filepath_or_buffer='crime.dta')
# origin_data.set_index(keys='state',inplace=True, drop=True)
origin_data.head(10)


# In[3]:


origin_data.loc[:,['crime','single','poverty']].describe()


# In[4]:


ols_model = smf.ols(formula='crime ~ poverty + single', data=origin_data)
ols_res = ols_model.fit()
print(ols_res.summary())


# In[5]:


rlm_model = smf.rlm(formula='crime ~ poverty + single', data=origin_data)
rlm_res = rlm_model.fit()
print(rlm_res.summary())


# In[6]:


rlm_res.weights.head(10)


# ## 样本影响图 ##
# 
# http://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.influence_plot.html
# 
# 可看出横轴为杠杆值（x取值），纵轴为学生化（类似于标准化）后的残差值（离群度），样本点的大小为影响度（如cook距离）

# In[7]:


fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
p1 = statsmodels.graphics.regressionplots.influence_plot(ols_res, ax=ax1)


# ## 误差平方 vs 杠杆值图 ##
# 
# http://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_leverage_resid2.html
# 
# 可看出横轴为残差平方值（离群度），纵轴为杠杆值（x取值）

# In[8]:


fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111)
p2 = statsmodels.graphics.regressionplots.plot_leverage_resid2(ols_res, ax=ax2)


# ## 去除异常点，做OLS ##
# 去除24号样本后，回归效果明显提升（去除50号后回归效果变差）

# In[9]:


filtered_data = origin_data.drop([24])


# In[10]:


# 去除50号后R-squared下降至不到0.5，去除24号后上升至0.78
ols_model_2 = smf.ols(formula='crime ~ poverty + single', data=filtered_data)
ols_res_2 = ols_model_2.fit()
print(ols_res_2.summary())

