#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


ipl_df=pd.read_csv('C:/Users/user/Desktop/Python/Machine Learning (Codes and Data Files)/Data/IPL IMB381IPL2013.csv')


# In[5]:


ipl_df.head(5)


# In[6]:


list(ipl_df.columns)


# In[7]:


ipl_df.info()


# In[8]:


ipl_df.COUNTRY.value_counts()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


sn.barplot(x='COUNTRY',y='SIXERS',data=ipl_df)


# In[11]:


sn.distplot(ipl_df['SOLD PRICE'])


# In[12]:


sn.boxplot(ipl_df['SOLD PRICE'])


# In[13]:


influencial_features=['AVE','SR-B','SIXERS','COUNTRY','PLAYING ROLE','ECON','CAPTAINCY EXP','SOLD PRICE']


# In[14]:


sn.heatmap(ipl_df[influencial_features].corr(),annot=True)


# In[15]:


ipl_df_int=ipl_df.drop(['Sl.NO.','PLAYER NAME','TEAM','BASE PRICE','AUCTION YEAR','SOLD PRICE'],axis=1)
X_features=list(ipl_df_int.columns)


# In[16]:


X_features=list(ipl_df_int.columns)


# In[17]:


categorical_features=['AGE','COUNTRY','PLAYING ROLE','CAPTAINCY EXP']


# In[18]:


ipl_encoded_df=pd.get_dummies(ipl_df[X_features],columns=categorical_features,drop_first=True)


# In[19]:


X_features=ipl_encoded_df.columns


# In[20]:


import statsmodels.api as sm
X=sm.add_constant(ipl_encoded_df)
Y=ipl_df['SOLD PRICE']


# In[36]:


Y=ipl_df['SOLD PRICE']


# In[37]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,train_size=0.8,random_state=42)


# In[23]:


ipl_model_1=sm.OLS(train_Y,train_X).fit()


# In[24]:


ipl_model_1.summary2()


# In[25]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors( X ):
    X_matrix = X.values
    vif = [ variance_inflation_factor( X_matrix, i ) for i in range( X_matrix.shape[1] ) ]
    vif_factors = pd.DataFrame()
    vif_factors['column'] = X.columns
    vif_factors['vif'] = vif
    return vif_factors


# In[28]:


vif_factors=get_vif_factors(X[X_features])
vif_factors


# In[29]:


columns_with_large_vif = vif_factors[vif_factors.vif > 4].column


# In[31]:


plt.figure( figsize = (12,10) )
sn.heatmap( X[columns_with_large_vif].corr(), annot = True );
plt.title( "Heatmap depicting correlation between features");


# In[32]:


columns_to_be_removed = ['T-RUNS', 'T-WKTS', 'RUNS-S', 'HS','AVE', 'RUNS-C', 'SR-B', 'AVE-BL','ECON', 'ODI-SR-B', 'ODI-RUNS-S', 'AGE_2', 'SR-BL']


# In[33]:


X_new_features = list( set(X_features) - set(columns_to_be_removed) )


# In[34]:


get_vif_factors( X[X_new_features] )


# In[38]:


train_X = train_X[X_new_features]
ipl_model_2 = sm.OLS(train_Y, train_X).fit()
ipl_model_2.summary2()


# In[40]:


significant_vars = ['COUNTRY_IND', 'COUNTRY_ENG', 'SIXERS', 'CAPTAINCY EXP_1']
train_X = train_X[significant_vars]
ipl_model_3 = sm.OLS(train_Y, train_X).fit()
ipl_model_3.summary2()


# In[42]:


def draw_pp_plot( model, title ):
  probplot = sm.ProbPlot( model.resid );
  plt.figure( figsize = (8, 6) );
  probplot.ppplot( line='45' );
  plt.title( title );
  plt.show();


# In[43]:


draw_pp_plot( ipl_model_3,"Normal P-P Plot of Regression Standardized Residuals");


# In[46]:


def get_standardized_values(vals):
    return ((vals-vals.mean())/vals.std())


# In[47]:


def plot_resid_fitted( fitted, resid, title):
  plt.scatter( get_standardized_values( fitted ),
  get_standardized_values( resid ) )
  plt.title( title )
  plt.xlabel( "Standardized predicted values")
  plt.ylabel( "Standardized residual values")
  plt.show()


# In[48]:


plot_resid_fitted( ipl_model_3.fittedvalues,ipl_model_3.resid,"Residual Plot")


# In[50]:


k = train_X.shape[1]
n = train_X.shape[0]
print( "Number of variables:", k, " and number of observations:", n)


# In[51]:


leverage_cutoff = 3*((k + 1)/n)
print( "Cutoff for leverage value: ", round(leverage_cutoff, 3) )


# In[52]:


from statsmodels.graphics.regressionplots import influence_plot
fig, ax = plt.subplots( figsize=(8,6) )
influence_plot( ipl_model_3, ax = ax )
plt.title( "Leverage Value Vs Residuals")
plt.show()


# In[55]:


ipl_df[ipl_df.index.isin( [23, 58, 83] )]


# In[57]:


train_X_new = train_X.drop( [23, 58, 83], axis = 0)
train_y_new = train_Y.drop( [23, 58, 83], axis = 0)


# In[60]:


ipl_model_4 = sm.OLS(train_y_new, train_X_new).fit()
ipl_model_4.summary2()


# In[61]:


draw_pp_plot( ipl_model_4,"Normal P-P Plot of Regression Standardized Residuals");


# In[62]:


pred_y = ipl_model_4.predict( test_X[train_X_new.columns])


# In[69]:


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(pred_y, test_Y))


# In[71]:


np.round( metrics.r2_score(pred_y, test_Y), 2 )


# In[102]:


ipl_df


# In[98]:


pred_y.columns=['SOLD PRCE PREDICTED']


# In[138]:


estimates_y=train_y_new.append(pred_y)


# In[144]:


ipl_final_df=pd.merge(ipl_df,estimates_y.rename('SOLD PRICE PREDICTED'), how='outer', left_index=True, right_index=True)


# In[145]:


ipl_final_df


# In[147]:


export_excel = ipl_final_df.to_excel (r'C:\Users\user\Desktop\Python\IPL_Auction_Prediction.xlsx', index = None, header=True)


# In[ ]:




