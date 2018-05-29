import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_feature-7.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train_data['type_label'] = type_in_train

num_class = train_data['type_label'].max() + 1
params = {
    'objective': 'multi:softmax',
    'eta': 0.01,
    'max_depth': 9,
    'subsample': 0.7,
    'lambda': 20,
    'gamma': 0.01,
    'eval_metric': 'merror',
    'seed': 0,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

# feature = [x for x in train.columns if x not in ['type_id', 'version_id', 'devnum_id', 'channel_id',
#                                                           'devstatus_id', 'datastatus_id', 'type_label']]
feature = [x for x in train_data.columns if x not in ['type_id', 'type_label', 'alarm_id']]

X_train, X_test, y_train, y_test = train_test_split(train_data[feature], train_data['type_label'].astype('int'),
                                                    test_size=0.2, random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

# xgbtrain = xgb.DMatrix(X_train, y_train)
# watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
# # model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)
model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)


# Because there are 8698 number of PRPS excel files for training, so we set 1500 number of PRPS excel files for testing
# In the rate of 5.8 : 1
print ("----predicting !----")

result = test_data
xgbtest = xgb.DMatrix(test_data[feature])
result['label'] = model.predict(xgbtest)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")
