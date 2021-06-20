import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import time
import lightgbm as lgb
import vaex
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

## FULL data for LGBM

Transfer to hdf5 in linux terminal

# data load
market = vaex.open('market_train_full.csv.hdf5').to_pandas_df()
market.set_index(market.columns[0], inplace = True)
market.index.name = ""

news = vaex.open('news_train_full.csv.hdf5').to_pandas_df()
news.set_index(news.columns[0], inplace = True)
news.index.name = ""
news.reset_index(drop = True, inplace = True)

#market = pd.read_csv("marketdata_sample.csv")
#news = pd.read_csv("news_sample.csv")

## LGBM Trainning

#news = pd.read_csv("news_sample.csv")
#market = pd.read_csv("marketdata_sample.csv")

# date type transform
def date_trans(df):
    for i in df.columns:
        if i == "time" or i == "sourceTimestamp" or i == "firstCreated":
            df[i] = pd.to_datetime(df[i])
    return df

market_train_df = date_trans(market)
news_train_df = date_trans(news)

news_train_df['date'] = news_train_df.time.dt.date

asset_code_dict = {k: v for v, k in enumerate(market_train_df['assetCode'].unique())}
drop_columns = [col for col in news_train_df.columns if col not in ['sourceTimestamp', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence', 'relevance','firstCreated', 'assetCodes']]
columns_news = ['firstCreated','relevance','sentimentClass','sentimentNegative','sentimentNeutral',
               'sentimentPositive','noveltyCount24H','noveltyCount7D','volumeCounts24H','volumeCounts7D','assetCodes','sourceTimestamp',
               'assetName','audiences', 'urgency', 'takeSequence', 'bodySize', 'companyCount', 
               'sentenceCount', 'firstMentionSentence','time']

def data_prep(market_df,news_df):
    market_df['date'] = market_df['time'].dt.date
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df.drop(['time'], axis=1, inplace=True)
    
    #news_df = news_df[columns_news]
    #news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    #news_df['firstCreated'] = pd.to_datetime(news_df.firstCreated)
    #news_df['firstCreated'] = news_df.firstCreated.dt.date
    #news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    #news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    #news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    #news_df['len_audiences'] = news_train_df['audiences'].map(lambda x: len(eval(x)))
    #kcol = ['firstCreated', 'assetCodes']
    #news_df = news_df.groupby(kcol, as_index=False).mean()
    #market_df = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
                            #right_on=['firstCreated', 'assetCodes'])
    #del news_df
    market_df['assetCodeT'] = market_df['assetCode'].map(asset_code_dict)
    #market_df = market_df.drop(columns = ['firstCreated','assetCodes','assetName']).fillna(0) 
    return market_df

# merge news data and market data 
market_train_df = data_prep(market_train_df, news_train_df)

num_columns = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
cat_columns = ['assetCodeT']
feature_columns = num_columns+cat_columns

# normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data_scaler = StandardScaler()
market_train_df[num_columns] = data_scaler.fit_transform(market_train_df[num_columns])

from sklearn.model_selection import train_test_split

market_train_df = market_train_df.reset_index()
#market_train_df = market_train_df.drop(columns='index')

# Random train-test split
train_indices, val_indices = train_test_split(market_train_df.index.values,test_size=0.1, random_state=92)

market_train_df.index

# Extract X and Y
def get_input(market_train, indices):
    X = market_train.loc[indices, feature_columns].values
    y = market_train.loc[indices,'returnsOpenNextMktres10'].map(lambda x: 0 if x<0 else 1).values
    #y = market_train.loc[indices,'returnsOpenNextMktres10'].map(lambda x: convert_to_class(x)).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'date']
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train_df, train_indices)
X_val,y_val,r_val,u_val,d_val = get_input(market_train_df, val_indices)

# Set up decay learning rate
def learning_rate_power(current_round):
    base_learning_rate = 0.19000424246380565
    min_learning_rate = 0.01
    lr = base_learning_rate * np.power(0.995,current_round)
    return max(lr, min_learning_rate)


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

tune_params = {'n_estimators': [200,500,1000,2500,5000],
              'max_depth': sp_randint(4,12),
              'colsample_bytree':sp_uniform(loc=0.8, scale=0.15),
              'min_child_samples':sp_randint(60,120),
              'subsample': sp_uniform(loc=0.75, scale=0.25),
              'reg_lambda':[1e-3, 1e-2, 1e-1, 1]}

fit_params = {'early_stopping_rounds':40,
              'eval_metric': 'accuracy',
              'eval_set': [(X_train, y_train), (X_val, y_val)],
              'verbose': 20,
              'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_power)]}

lgb_clf = lgb.LGBMClassifier(n_jobs=4, objective='binary',random_state=1)
gs = RandomizedSearchCV(estimator=lgb_clf, 
                        param_distributions=tune_params, 
                        n_iter=2,
                        scoring='f1',
                        cv=5,
                        refit=True,
                        random_state=1,
                        verbose=True)

lgb_clf = lgb.LGBMClassifier(n_jobs=4,
                             objective='multiclass',
                            random_state=100)

opt_params = {'n_estimators':500,
              'boosting_type': 'dart',
              'objective': 'binary',
              'num_leaves':2452,
              'min_child_samples':212,
              'reg_lambda':0.01}
lgb_clf.set_params(**opt_params)
lgb_clf.fit(X_train, y_train, **fit_params)

from sklearn.metrics import accuracy_score

accuracy_score(y_val, lgb_clf.predict(X_val))

# Rescale confidence
def rescale(data_in, data_ref):
    scaler_ref =  StandardScaler()
    scaler_ref.fit(data_ref.reshape(-1,1))
    scaler_in = StandardScaler()
    data_in = scaler_in.fit_transform(data_in.reshape(-1,1))
    data_in = scaler_ref.inverse_transform(data_in)[:,0]
    return data_in

y_pred_proba = lgb_clf.predict_proba(X_val)
predicted_return = y_pred_proba[:,1] - y_pred_proba[:,0]
predicted_return = rescale(predicted_return, r_train)


r_val = r_val.clip(-1,1) 
x_t_i = lgb_clf.predict(X_val) * r_val * u_val
data = {'day' : d_val, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print('Validation score', score_valid)


y_predict = lgb_clf.predict(X_val)

confidence_valid = y_predict[:]*2 -1 
#print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()