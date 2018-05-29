import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

# feature = [x for x in train_data.columns if x not in ['type_id', 'alarm_id', 'cycle_num']]
# min_max_scaler = preprocessing.MinMaxScaler()
# train_scaled = pd.DataFrame(min_max_scaler.fit_transform(train_data[feature]))
# train_scaled = pd.concat((train_data[['type_id', 'alarm_id', 'cycle_num']], train_scaled), axis=1)
# test_scaled = pd.DataFrame(min_max_scaler.fit_transform(test_data[feature]))
# test_scaled = pd.concat((test_data[['type_id', 'alarm_id', 'cycle_num']], test_scaled), axis=1)


train_data['is_train'] = 1
test_data['is_train'] = 0
data = (pd.concat((train_data, test_data), axis=0)).reset_index()

feature = [x for x in data.columns if x not in ['type_id', 'alarm_id', 'cycle_num', 'is_train']]
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = pd.DataFrame(min_max_scaler.fit_transform(data[feature]))
data_scaled = pd.concat((data[['type_id', 'alarm_id', 'cycle_num', 'is_train']], data_scaled), axis=1)

train_scaled = data_scaled[data_scaled['is_train'] == 1].reset_index()
test_scaled = data_scaled[data_scaled['is_train'] == 0].reset_index()

mainLabelEncoder = LabelEncoder()
type_label = train_scaled[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_scaled[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train = train_scaled
train['type_label'] = type_in_train
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
feature = [x for x in train.columns if x not in ['index', 'type_id', 'is_train', 'type_label']]

# xgbtrain = xgb.DMatrix(data_for_train[feature], data_for_train['type_id'])
# xgbeval = xgb.DMatrix(data_for_valid[feature], data_for_valid['type_id'])
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

xgbtrain = xgb.DMatrix(train[feature], train['type_label'].astype('int'))
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)
model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)

# Because there are 8698 number of PRPS excel files for training, so we set 1500 number of PRPS excel files for testing
# In the rate of 5.8 : 1
print ("----predicting !----")

result = test_scaled
xgbtest = xgb.DMatrix(test_scaled[feature])
result['label'] = model.predict(xgbtest)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result_data_scaled.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")
