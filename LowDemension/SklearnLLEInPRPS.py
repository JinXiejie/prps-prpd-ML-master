import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import manifold
import xgboost as xgb
from enum import Enum

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

feature = [x for x in train_data.columns if x not in ['type_id']]
train_data_lle = manifold.LocallyLinearEmbedding(n_neighbors=180, n_components=17,
                                                 method='hessian').fit_transform(train_data)
test_data_lle = manifold.LocallyLinearEmbedding(n_neighbors=180, n_components=17,
                                                method='hessian').fit_transform(test_data)

train_data_lle = pd.DataFrame(train_data_lle)
test_data_lle = pd.DataFrame(test_data_lle)

train_data_lle = pd.concat((train_data['type_id'], train_data_lle), axis=1)

test_data_lle = pd.concat((test_data['type_id'], test_data_lle), axis=1)

mainLabelEncoder = LabelEncoder()
type_label = train_data_lle[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data_lle[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train = train_data_lle
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
result = test_data_lle
# feature = [x for x in test_data_pca.columns if x not in ['type_id']]
xgbtest = xgb.DMatrix(test_data_lle[feature])
result['label'] = model.predict(xgbtest)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result_LLE.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")
