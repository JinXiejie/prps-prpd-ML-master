import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')


def zero_mean(data_matrix):
    mean_val = np.mean(data_matrix, axis=0)
    new_data = data_matrix - mean_val
    return new_data, mean_val


def pca(data_matrix, n):
    new_data, mean_val = zero_mean(data_matrix)
    cov_matrix = np.cov(new_data, rowvar=False)

    eig_vals, eig_vectors = np.linalg.eig(np.mat(cov_matrix))

    eig_val_sorted = np.argsort(eig_vals)
    n_eig_val_sorted = eig_val_sorted[-1:-(n + 1):-1]
    n_eig_vectors = eig_vectors[:, n_eig_val_sorted]
    low_demension_data_matrix = new_data * n_eig_vectors

    # reconsitution the data
    recon_data = (low_demension_data_matrix * n_eig_vectors.T) + mean_val
    return low_demension_data_matrix, recon_data


feature = [x for x in train_data.columns if x not in ['type_id']]
high_matrix_train = np.matrix(train_data[feature])
low_matrix_train, reconsitution_data_train = pca(high_matrix_train, 68)
train_data_pca = pd.DataFrame(low_matrix_train)
train_data_pca = pd.concat((train_data['type_id'], train_data_pca), axis=1)

high_matrix_test = np.matrix(test_data[feature])
low_matrix_test, reconsitution_data_test = pca(high_matrix_test, 68)
test_data_pca = pd.DataFrame(low_matrix_test)
test_data_pca = pd.concat((test_data['type_id'], test_data_pca), axis=1)

mainLabelEncoder = LabelEncoder()
type_label = train_data_pca[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data_pca[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train = train_data_pca
train['type_label'] = type_in_train

num_class = train['type_id'].max() + 1
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
result = test_data_pca
# feature = [x for x in test_data_pca.columns if x not in ['type_id']]
xgbtest = xgb.DMatrix(test_data_pca[feature])
result['label'] = model.predict(xgbtest)
result['label'] = result['label'].apply(lambda x: int(x))
result['label'] = mainLabelEncoder.inverse_transform(result['label'])
dest_file = 'ResultData/result_PCA.csv'
result.to_csv(dest_file, index=None)
print ("----over !----")
