
# coding: utf-8

# # 多重共线性判断与处理 Multicollinearity #

# ### 数据集 ### 
# <b>1994年—2007年中国旅游收入及相关数据 </b>
# 
# <li>年份</li>
# <li>国内旅游收入Y（亿元）</li>
# <li>国内旅游人数X2（万人次）</li>
# <li>城镇居民人均旅游花费X3（元）</li>
# <li>农村居民人均旅游花费X4 （元）</li>
# <li>公路里程 X5（万km）</li>
# <li>铁路里程X6（万km）</li>

# In[1]:


import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


origin_data = pd.read_csv('multicollinearity_demo.csv',
                          skiprows=[0,1],
                          names=['year', 'income', 'people', 'consume_city', 'consume_country', 'miles_road', 'miles_rail'])


# ## 探究性分析 ##

# ### 绘制相关矩阵 ###

# In[3]:


pd.plotting.scatter_matrix(origin_data, figsize=(12,12))


# ## STEPWISE 前向逐步分析法 ##

# In[4]:


# 向前法
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula = "{} ~ {}".format(
                response,' + '.join(selected + [candidate]))
            print('current formula: {}'.format(formula))
            
            # 可以用任何模型如glm, ols, wls等， 选择适合当前业务的
            aic = smf.ols(formula=formula, data=data).fit().aic
            aic_with_candidates.append((aic, candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score, best_candidate=aic_with_candidates.pop()
        
        if current_score > best_new_score: 
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print ('variable is {}, aic is {:.8}, continuing..\n'.format(selected, current_score))
        else:        
            print ('forward selection over!')
            break

    formula = "{} ~ {} ".format(response,' + '.join(selected))
    print('final formula is {}'.format(formula))
    
    # 可以用任何模型如glm, ols, wls等， 选择适合当前业务的
    model = smf.ols(formula=formula, data=data).fit()
    
    # 对照模型，不使用STEPWISE
    model_contrast = smf.ols(formula='income ~ + year + people + consume_city + consume_country + miles_road + miles_rail', data=data).fit()
    
    return model, model_contrast


# In[5]:


# 样本量必须大于等于20
# UserWarning: kurtosistest only valid for n>=20

data_for_select = origin_data
model_stepwise, model_contrast = forward_select(data=data_for_select, response='income')


# In[6]:


# stepwise选择变量后的模型
model_stepwise.summary()


# In[7]:


# 对照模型
model_contrast.summary()


# ## 方差膨胀因子VIF ##
# 每个自变量（外生变量, X）与其他自变量做OLS，获得其他自变量对于该自变量的R^2，R^2越大，VIF增大，说明该自变量存在被（线性）替代的可能

# ### 分别对旅游人数people, 公路里程miles_road, 铁路里程miles_rail变量、STEPWISE选择后的变量计算其VIF膨胀因子 ###
# 参考 https://blog.csdn.net/songhao22/article/details/79369950

# In[8]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

data_choosed = origin_data.loc[:,['people', 'miles_road', 'miles_rail']]
for i, name in enumerate(data_choosed.columns):
    vif = variance_inflation_factor(np.array(data_choosed), i)    # 针对第i列变量，计算其他变量对其的VIF，若过大则说明其他变量与其存在近似线性关系
    print( '{: <20}'.format(name + '"\'s VIF:'), '{:.6}'.format(vif))
    # '格式'.format(变量) 输出法，.6表示保留六位长度，<^>为左中右对齐，20为强制长度


# In[9]:


data_stepwised = origin_data.loc[:,['people', 'consume_city', 'consume_country', 'miles_rail']]
for i, name in enumerate(data_stepwised.columns):
    vif = variance_inflation_factor(np.array(data_stepwised), i)    # 针对第i列变量，计算其他变量对其的VIF，若过大则说明其他变量与其存在近似线性关系
    print( '{: <20}'.format(name + '"\'s VIF:'), '{:.6}'.format(vif))
    # '格式'.format(变量) 输出法，.6表示保留六位长度，<^>为左中右对齐，20为强制长度


# ### 注：VIF较大只是存在多重共线性的必要条件而非充分条件。即，若出现多重共线性，则VIF大，VIF大并不一定能推出一定存在多重共线性，只作为参考和验证###

# In[10]:


np.random.seed(64)
demo_x = np.random.rand(4,4)

# 对于numpy定义的行变量，需转置后，针对列变量进行VIF检验，若是DataFrame则不需要
vif = variance_inflation_factor(demo_x.T,1)

print("A Gauss Distribution Matrix's VIF:", vif)

