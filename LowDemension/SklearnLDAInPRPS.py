import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

feature = [x for x in train_data.columns if x not in ['type_id']]

lda = LinearDiscriminantAnalysis(n_components=4)
lda.fit(train_data[feature], train_data['type_id'])
train_data_lda = lda.transform(train_data[feature])
lda.fit(test_data[feature], test_data['type_id'])
test_data_lda = lda.transform(test_data[feature])

train_data_lda = pd.DataFrame(train_data_lda)
test_data_lda = pd.DataFrame(test_data_lda)

train_data_lda = pd.concat((train_data['type_id'], train_data_lda), axis=1)

test_data_lda = pd.concat((test_data['type_id'], test_data_lda), axis=1)

mainLabelEncoder = LabelEncoder()
type_label = train_data_lda[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data_lda[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train = train_data_lda
train['type_label'] = type_in_train
# feature = [x for x in train.columns if x not in ['type_id', 'type_label']]
# data_for_train, data_for_valid, label_for_train, label_for_valid = train_test_split(train[feature],
#                                                                                     train['type_label'],
#                                                                                     test_size=0.2)

# num_class = train['type_id'].max() + 1
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
feature = [x for x in train.columns if x not in ['type_id', 'type_label']]

# xgbtrain = xgb.DMatrix(data_for_train, label_for_train)
# xgbeval = xgb.DMatrix(data_for_valid, label_for_valid)
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

xgbtrain = xgb.DMatrix(train[feature], train['type_label'].astype('int'))
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]

# model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)
model = xgb.train(params, xgbtrain, num_boost_round=200, early_stopping_rounds=20, evals=watchlist)

# Because there are 8698 number of PRPS excel files for training, so we set 1500 number of PRPS excel files for testing
# In the rate of 5.8 : 1
print ("----predicting !----")
result = test_data_lda
# feature = [x for x in test_data_pca.columns if x not in ['type_id']]
xgbtest = xgb.DMatrix(test_data_lda[feature])
result['label'] = model.predict(xgbtest)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result_LDA.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")
