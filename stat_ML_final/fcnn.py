import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse
from keras.callbacks import EarlyStopping, ModelCheckpoint
import gc
import vaex

print("Load package complete.")

import resource
def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

limit_memory(1024*1024*1024*540)

# data load
market = vaex.open('market_train_full.csv.hdf5').to_pandas_df()
market.set_index(market.columns[0], inplace = True)
market.index.name = ""

news = vaex.open('news_train_full.csv.hdf5').to_pandas_df()
news.set_index(news.columns[0], inplace = True)
news.index.name = ""
news.reset_index(drop = True, inplace = True)

print("Data loader complete.")

dfm = market  
dfn = news

def date_trans(df):
    for i in df.columns:
        if i == "time" or i == "sourceTimestamp" or i == "firstCreated":
            df[i] = pd.to_datetime(df[i])
    return df

dfm = date_trans(dfm)
dfn = date_trans(dfn)

print("Date type transfer complete")


def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train
news_train = preprocess_news(dfn)

print("news preprocessing complete.")

def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df

#因為要根據asset_code來merge
index_df = unstack_asset_codes(news_train)

print("unstack asset coder complete.")

def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

news_unstack = merge_news_on_index(news_train, index_df)
del news_train, index_df
gc.collect()

def group_news(news_frame):
    news_frame['date'] = pd.to_datetime(news_frame.time).dt.date  # Add date column
    
    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date','assetName']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date','assetName']}
    return gp.astype(float_cols)

news_agg1 = group_news(news_unstack)
drop_list = [
        'noveltyCount12H_mean', 'noveltyCount24H_mean', 'noveltyCount3D_mean', 'noveltyCount5D_mean', 
    'noveltyCount7D_mean', 'sentimentClass_mean', 'marketCommentary_mean',
'headlineTag_mean', 'provider_mean', 'takeSequence_mean', 'urgency_mean' 
    ]
news_agg1.drop(drop_list, axis=1, inplace=True)
#del news_unstack; gc.collect()
 #169 group by assetCode and mean



news_agg2 = group_news(news_unstack)
news_agg2 = news_agg2[[ 'assetCode', 'date', 'assetName',
        'noveltyCount12H_mean', 'noveltyCount24H_mean', 'noveltyCount3D_mean', 'noveltyCount5D_mean', 
    'noveltyCount7D_mean', 'sentimentClass_mean', 'marketCommentary_mean', 'provider_mean', 'takeSequence_mean'
    ]]
#del news_unstack; gc.collect()
news_agg2 = news_agg2.round() #169 group by assetCode and mean
cat_col = news_agg2.select_dtypes(include='float32').columns.to_list()
news_agg2[cat_col] = news_agg2[cat_col].astype('int').astype('category')
news_agg = news_agg1.merge(news_agg2, how='left', on=['assetCode', 'date','assetName'])
dfm['date'] = pd.to_datetime(dfm.time).dt.date
df = dfm.merge(news_agg, how='left', on=['assetCode', 'date'])


print("dataframe merge complete.")

cat_cols = ['assetCode', 'noveltyCount12H_mean', 'sentimentClass_mean', 'marketCommentary_mean', 'takeSequence_mean', ]
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10','bodySize_mean','sentenceCount_mean','wordCount_mean',
           'relevance_mean', 'volumeCounts12H_mean']

train_indices, val_indices = train_test_split(df.index.values,test_size=0.25, random_state=23)
def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(df.loc[train_indices, cat].astype(str).unique())}
    df[cat] = df[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets

df[num_cols] = df[num_cols].fillna(0)

print("Catergory variables encode complete.")

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[5], name=cat))

#categorical_embeddings = []
#for i, cat in enumerate(cat_cols):
#    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)

#numerical_inputs = Input(shape=(11,), name='num')
numerical_inputs = Input(shape=(16,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)

print("Model construction complete.")


print(model.summary())


def get_input(df, indices):
    X_num = df.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = df.loc[indices, cat_cols].values
    y = (df.loc[indices,'returnsOpenNextMktres10'] >= 0).values #為什麼要>=0? 因為分類
    r = df.loc[indices,'returnsOpenNextMktres10'].values
    u = df.loc[indices, 'universe']
    d = pd.to_datetime(df.loc[indices, 'time']).dt.date
    return X,y,r,u,d


# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(df, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(df, val_indices)

print("dataset splitting complete.")


check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=2,
          verbose=True,
          callbacks=[early_stop,check_point]) 

print("Model fitting done.")


model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1 #為什麼要*2-1?
print("Accuracy score")
print(accuracy_score(confidence_valid>0,y_valid))


r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print("Score")
print(score_valid) 
