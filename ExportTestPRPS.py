import pandas as pd
import os
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

# train data
train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train1.csv')
train_data['cycle_num'] = train_data.count(axis=0)[0]
feature = [x for x in train_data.columns if
           x not in ['version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                     'alarm_id']]
train_data = train_data.groupby(['type_id', 'alarm_id'])[feature].agg(['mean', 'median', 'max', 'min']).reset_index()

for i in range(1, 10161):
    filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train' + str(i) + '.csv'
    if os.path.exists(filePath):
        temp = pd.read_csv(filePath)
        temp['cycle_num'] = temp.count(axis=0)[0]
        temp = temp.groupby(['type_id', 'alarm_id'])[feature].agg(['mean', 'median', 'max', 'min']).reset_index()
        train_data = pd.concat((train_data, temp), axis=0)

train_data = train_data.reset_index()
train_data.drop(['index'], axis=1, inplace=True)
train_data.to_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv', index=False)


# test data
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train1.csv')
test_data['cycle_num'] = test_data.count(axis=0)[0]
feature = [x for x in test_data.columns if
           x not in ['version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                     'alarm_id']]

test_data = test_data.groupby(['type_id', 'alarm_id'])[feature].agg(['mean', 'median', 'max', 'min']).reset_index()
for i in range(1, 1500):
    filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/prps_train' + str(i) + '.csv'
    if os.path.exists(filePath):
        temp = pd.read_csv(filePath)
        temp['cycle_num'] = temp.count(axis=0)[0]
        temp = temp.groupby(['type_id', 'alarm_id'])[feature].agg(['mean', 'median', 'max', 'min']).reset_index()
        test_data = pd.concat((test_data, temp), axis=0)
# delete the column of which value is NAN
# data.drop(['Unnamed: 8'], axis=1, inplace=True)
test_data = test_data.reset_index()
test_data.drop(['index'], axis=1, inplace=True)
test_data.to_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv', index=False)


# def divide_into_seven_part(index_num):
#     index_num = int(index_num % 7)
#     return index_num

# data['part'] = data['index']
# data['part'] = data.part.apply(divide_into_seven_part)
# data.drop(['index'], axis=1, inplace=True)
# data_for_train = data[data['part'] > 0]
# data_for_test = data[data['part'] == 0]
# data_for_train.to_csv('E:/JinXiejie/data/TrainData-PDMSystemPdmSys_CouplerSPDC-Channel_2/TrainData/data_for_train.csv',
#                       index=False)
# data_for_test.to_csv('E:/JinXiejie/data/TrainData-PDMSystemPdmSys_CouplerSPDC-Channel_2/TrainData/data_for_test.csv',
#                      index=False)
