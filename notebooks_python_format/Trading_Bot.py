#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
from datetime import datetime
from sklearn.model_selection import train_test_split


# In[2]:


ticker = 'NFLX'
benchmark = 'SPY'
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
data = yf.download([ticker,benchmark], start, end)

df = pd.DataFrame(index = data.index)
data.head(5)


# In[3]:


# Calculate EMA (Exponential Moving Average)
df['EMA10'] = ta.trend.ema_indicator(data['Close', ticker], window=10)
df['EMA30'] = ta.trend.ema_indicator(data['Close', ticker], window=30)

# Calculate ATR (Average True Range)
df['ATR'] = ta.volatility.average_true_range(data['High', ticker], data['Low', ticker], data['Close', ticker], window=14)

# Calculate ADX (Average Directional Index)
df['ADX'] = ta.trend.adx(data['High', ticker], data['Low', ticker], data['Close', ticker], window=14)

# Calculate RSI (Relative Strength Index)
df['RSI'] = ta.momentum.rsi( data['Close', ticker], window=14)

# Calculate MACD (Moving Average Convergence Divergence)
df['MACD'] = ta.trend.macd( data['Close', ticker], window_slow=26, window_fast=12)

# Calculate MACD Signal line
df['MACD_signal'] = ta.trend.macd_signal( data['Close', ticker], window_slow=26, window_fast=12, window_sign=9)

df.shape[0]


# # Introduce features

# In[4]:


df['Close_EMA_10'] = np.where(data['Close', ticker]> df['EMA10'], 1, -1)
df['EMA_10_EMA_30'] = np.where(df['EMA10'] > df['EMA30'], 1, -1)
df['MACD_Signal_MACD'] = np.where(df['MACD_signal'] > df['MACD'], 1, -1)

df.tail(11)


# <h2>Creating the target variables</h2>
# This are the variables whose values are to be modeled and predicted by othervariables. There must be one and only one target variable in a decision tree analysis.
# The target variable for the classification algorithm also uses the lagged return, butbecause the output is categorical, we must transform it. If the return was positive,we assign 1 and if it was negative, we assign 0.

# In[5]:


df['returned'] = np.log(data['Close', ticker]/data['Close', ticker].shift(1))
df['return_'+benchmark] = np.log(data['Close', benchmark]/data['Close', benchmark].shift(1))
df['target'] = np.where(df['returned'] > 0, 1, 0)

df


# In[6]:


X = df[['ATR', 'ADX','RSI', 'Close_EMA_10', 'EMA_10_EMA_30', 'MACD_Signal_MACD']] #features
y = df.target #target

X_clean = X.dropna()
Y_clean = y[X.index.isin(X_clean.index)]

X=X_clean
y=Y_clean

#splitting the data and building the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# In[7]:


df.describe()


# Import all machine learning models

# In[8]:


from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


# In[9]:


def plot_models(results, names):
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# In[10]:


seed = 7
models = []

models.append(('LogisticRegression', LogisticRegression(random_state=seed)))
models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
models.append(('ExtraTreesClassifier',ExtraTreesClassifier(random_state=seed)))
models.append(('AdaBoostClassifier',AdaBoostClassifier(DecisionTreeClassifier(random_state=seed),random_state=seed,learning_rate=0.1)))
models.append(('SVM',svm.SVC(random_state=seed)))
models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=seed)))
models.append(('MLPClassifier',MLPClassifier(random_state=seed)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #print(msg) 

#print(results, names)
plot_models(results, names)


# # Hyper-parameter Tuning

# In[11]:


param_grid={
    'n_estimators':[100,200,300],
    'max_depth':[10,20,30],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}


# In[12]:


#for name, model in models:
    #grid_search = GridSearchCV(estimator = model, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
grid_search = GridSearchCV(estimator = RandomForestClassifier(random_state=42), param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)

best_params=grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")
best_model=RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train,y_train)


# # Performance Evaluation

# In[13]:


y_pred=best_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[14]:


accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")


# In[15]:


df.head(5)


# In[16]:


df.columns


# Plot graph to check which strategy is working

# In[17]:


train_length=0
#plt.plot((df.strategy_returns[train_length:]+1).cumprod(),'b-',label='Strategy returns decision tree ')
plt.plot((df.returned[train_length:]+1).cumprod(),'g-',label='Strategy returns Buy and Hold ')
plt.plot((df['return_'+benchmark][train_length:]+1).cumprod(),'r-',label=benchmark+' returns Buy and Hold ')
plt.ylabel('Strategy returns(%)')
plt.legend()
plt.show()


# In[ ]:




