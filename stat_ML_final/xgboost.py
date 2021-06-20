import pandas as pd
import numpy as np
from sklearn import *
from xgboost import XGBClassifier
import time
from sklearn.metrics import accuracy_score

#Include News Information
market_train = pd.read_csv('D:/博士生课程/统计学习理论与运用/sample/marketdata_sample.csv')
news_train = pd.read_csv('D:/博士生课程/统计学习理论与运用/sample/news_sample.csv')

def data_prep(market_train,news_train):
    market_train['time'] = pd.to_datetime(market_train['time']).dt.date
    news_train.time = pd.to_datetime(news_train.time).dt.hour
    news_train.sourceTimestamp= pd.to_datetime(news_train.sourceTimestamp).dt.hour
    news_train.firstCreated = pd.to_datetime(news_train.firstCreated).dt.date
    news_train['assetCodesLen'] = 0
    for i in range(len(news_train)):
        news_train.loc[i,'assetCodesLen'] = len(eval(news_train.loc[i,'assetCodes']))
        news_train.loc[i,'assetCodes'] = list(eval(news_train.loc[i,'assetCodes']))[0]
    keycol = ['firstCreated', 'assetCodes']
    news_train = news_train.groupby(keycol, as_index=False).mean()
    market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    return market_train

market_train = data_prep(market_train,news_train)

up = market_train.returnsOpenNextMktres10 >= 0

fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
u = market_train.universe
r = market_train.returnsOpenNextMktres10.values 
d = market_train['time_x']

mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

assert X.shape[0] == up.shape[0] == r.shape[0] == u.shape[0] == d.shape[0]

Indices_train,Indices_test=\
model_selection.train_test_split(market_train.index.values,test_size=0.25, random_state=99)

X_train = X[Indices_train]
X_test = X[Indices_test]
up_train = up[Indices_train]
up_test = up[Indices_test]
u_train = u[Indices_train]
u_test = u[Indices_test]
r_train = r[Indices_train]
r_test = r[Indices_test]
d_train = d[Indices_train]
d_test = d[Indices_test]

! pip install xgboost

xgb_up = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)

t = time.time()
print('Fitting Up')
xgb_up.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}')

accuracy_score(xgb_up.predict(X_test),up_test)

r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = up_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)

#Market Only

market_train = pd.read_csv('D:/博士生课程/统计学习理论与运用/sample/marketdata_sample.csv')
news_train = pd.read_csv('D:/博士生课程/统计学习理论与运用/sample/news_sample.csv')

def data_prep(market_train,news_train):
    market_train['time'] = pd.to_datetime(market_train['time']).dt.date
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    return market_train

market_train = data_prep(market_train,news_train)

up = market_train.returnsOpenNextMktres10 >= 0

fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
u = market_train.universe
r = market_train.returnsOpenNextMktres10.values 
d = market_train['time']

assert X.shape[0] == up.shape[0] == r.shape[0] == u.shape[0] == d.shape[0]

mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

Indices_train,Indices_test=\
model_selection.train_test_split(market_train.index.values,test_size=0.25, random_state=99)

X_train = X[Indices_train]
X_test = X[Indices_test]
up_train = up[Indices_train]
up_test = up[Indices_test]
u_train = u[Indices_train]
u_test = u[Indices_test]
r_train = r[Indices_train]
r_test = r[Indices_test]
d_train = d[Indices_train]
d_test = d[Indices_test]

xgb_up = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)

t = time.time()
print('Fitting Up')
xgb_up.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}')

accuracy_score(xgb_up.predict(X_test),up_test)

r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = up_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)