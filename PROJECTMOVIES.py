#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

df = pd.read_csv(r'C:\Users\Rhythm\Documents\MOVIES.csv')


# In[3]:


df.head()


# In[5]:


for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pct_missing))


# In[7]:


df.dtypes


# In[15]:


df.replace(np.NaN,0)
df


# In[17]:


df['budget'] = df['budget'].astype('int')
df['gross'] = df['gross'].astype('int')
df


# In[20]:


df['yearcorrect'] = df['released'].astype(str).str[:4]
df


# In[21]:


df.sort_values(by=['gross'], inplace = False, ascending = False)


# In[23]:


pd.set_option('display.max_rows', None)
df


# In[26]:


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[29]:


sns.regplot(x="gross", y="budget", data=df, scatter_kws={"color":"blue"}, line_kws={"color":"green"})


# In[32]:


df.corr(method ='pearson')


# In[33]:


df.corr(method ='kendall')


# In[35]:


correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[36]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[37]:


correlation_matrix = df_numerized.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[ ]:


CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[ ]:





# In[ ]:




