import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum
import operator
from matplotlib import pylab as plt

# train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train.csv')


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


train_data['is_train'] = 1
test_data['is_train'] = 0
data = (pd.concat((train_data, test_data), axis=0))
data = pd.concat((data, pd.get_dummies(data['alarm_id'])), axis=1)
train_data = data[data['is_train'] == 1].reset_index()
test = data[data['is_train'] == 0].reset_index()

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train = train_data
train['type_label'] = type_in_train
train.drop(['index'], axis=1, inplace=True)

num_class = train['type_label'].max() + 1
params = {
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': 9,
    'subsample': 0.7,
    'eval_metric': 'merror',
    'seed': 0,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

# feature = [x for x in train.columns if x not in ['type_id', 'version_id', 'devnum_id', 'channel_id',
#                                                           'devstatus_id', 'datastatus_id', 'type_label']]
feature = [x for x in train.columns if x not in ['type_id', 'type_label', 'is_train', 'alarm_id']]

# xgbtrain = xgb.DMatrix(data_for_train[feature], data_for_train['type_id'])
# xgbeval = xgb.DMatrix(data_for_valid[feature], data_for_valid['type_id'])
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

xgbtrain = xgb.DMatrix(train[feature], train['type_label'].astype('int'))
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)
model = xgb.train(params, xgbtrain, num_boost_round=7000, evals=watchlist)

ceate_feature_map(feature)

importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(16, 30))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')
