import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.cross_validation import train_test_split

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')
# a = pd.read_csv('E:/JinXiejie/PythonCases/PyDemo/ResultData/result_binary.csv')

train_data['is_train'] = 1
test_data['is_train'] = 0
data = (pd.concat((train_data, test_data), axis=0)).reset_index()

feature = [x for x in data.columns if x not in ['type_id', 'alarm_id', 'cycle_num', 'is_train']]
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = pd.DataFrame(min_max_scaler.fit_transform(data[feature]))
data_scaled = pd.concat((data[['type_id', 'alarm_id', 'cycle_num', 'is_train']], data_scaled), axis=1)

train_data = data_scaled[data_scaled['is_train'] == 1].reset_index()
test_data = data_scaled[data_scaled['is_train'] == 0].reset_index()

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'max_depth': 9,
    'subsample': 0.7,
    # 'eval_metric': 'logloss',
    'eval_metric': 'error',
    'seed': 0,
    'missing': -999,
    'silent': 1,
}

type_list = list(set(list(data_scaled['type_id'])))
models = []
for type_id in type_list:
    print ("----------------------- " + str(type_id) + " -----------------------")
    posi_data = train_data[train_data['type_id'] == type_id]
    posi_data['type_label'] = 1
    neg_data = train_data[train_data['type_id'] != type_id]
    neg_data['type_label'] = 0
    train = pd.concat((posi_data, neg_data), axis=0)

    feature = [x for x in train.columns if x not in ['index', 'type_id', 'is_train', 'type_label']]

    train, valid = train_test_split(train, test_size=0.2)
    xgbtrain = xgb.DMatrix(train[feature], train['type_label'])
    xgbeval = xgb.DMatrix(valid[feature], valid['type_label'])
    watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

    # model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)
    model = xgb.train(params, xgbtrain, num_boost_round=1000, early_stopping_rounds=30, evals=watchlist)
    models.append(model)
    print ("----------------------- " + str(type_id) + " -----------------------")


print ("----predicting !----")

result = test_data
xgbtest = xgb.DMatrix(test_data[feature])
for model in models:
    # model = models[i]
    result = pd.concat((result, pd.DataFrame(model.predict(xgbtest))), axis=1)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")

