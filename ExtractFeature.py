import pandas as pd
import os
import math
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum


def extract_feature(train_data):
    feature = [x for x in train_data.columns if
               x not in ['version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                         'alarm_id']]
    cycle_num = train_data.count()[0]
    mean_data = train_data.groupby(['type_id', 'alarm_id'])[feature].agg('mean').reset_index()
    train_data = train_data[feature]
    column_nums = train_data.count(axis=1)[0]
    column_nums_half = column_nums / 2
    epsilon = 0.0001
    result = pd.DataFrame()

    result = pd.concat((result, mean_data), axis=1)
    posi_phase = []
    for i in range(column_nums_half):
        posi_phase.append('p' + str(i))

    neg_phase = []
    for i in range(column_nums_half, column_nums):
        neg_phase.append('p' + str(i))
    posi_phase_feature = [x for x in train_data.columns if x in posi_phase]
    neg_phase_feature = [x for x in train_data.columns if x in neg_phase]

    # feature 1: posi_neg_q
    # feature of balance between positive q in positive phase and negative q in negative phase
    posi_num = 0
    posi_sum = 0
    neg_num = 0
    neg_sum = 0
    for index, row in train_data.iterrows():
        for column in train_data[posi_phase_feature].columns:
            if row[column] > 0:
                posi_num += 1
                posi_sum += row[column]
        for column in train_data[neg_phase_feature].columns:
            if row[column] > 0:
                neg_num += 1
                neg_sum += row[column]
    # add a very small number epsilon in case either neg_num or posi_sum is zero
    posi_neg_q = (neg_sum * posi_num) / (neg_num * posi_sum + epsilon)
    temp = []
    temp.append(posi_neg_q)
    result['posi_neg_q'] = pd.DataFrame(temp)

    # feature 2: first_phase_difference
    # rate in positive phase and negative phase of the first q
    first_posi = 1
    first_neg = 33
    for index, row in train_data.iterrows():
        for column in train_data[posi_phase_feature].columns:
            if row[column] > 0:
                first_posi = int(column.split('p')[1]) + 1
                break
        for column in train_data[neg_phase_feature].columns:
            if row[column] > 0:
                first_neg = int(column.split('p')[1]) + 1
                break
    first_phase_difference = float(first_posi) - float(first_neg)
    temp = []
    temp.append(first_phase_difference)
    result['first_phase_difference'] = pd.DataFrame(temp)

    # feature 3: posi_phase_width
    # the width of PD in positive phase
    # feature 4: neg_phase_width
    # the width of PD in negative phase
    posi_phase_width = 0
    neg_phase_width = 0

    for column in train_data[posi_phase_feature].columns:
        nzero_column = train_data[train_data[column] > 0].count()[0]
        if float(nzero_column) / float(cycle_num) > 0.1:
            posi_phase_width += 1
    for column in train_data[neg_phase_feature].columns:
        nzero_column = train_data[train_data[column] > 0].count()[0]
        if float(nzero_column) / float(cycle_num) > 0.1:
            neg_phase_width += 1
    temp = []
    temp.append(posi_phase_width)
    result['posi_phase_width'] = pd.DataFrame(temp)
    temp = []
    temp.append(neg_phase_width)
    result['neg_phase_width'] = pd.DataFrame(temp)

    # feature 5: phase_mean
    # the mean of all phases
    phase_mean = 0.0
    phase_nzero = 0
    for column in train_data.columns:
        phase_nzero += train_data[train_data[column] > 0].count()[0]
    for column in train_data.columns:
        phase_mean += float(math.pow(train_data[train_data[column] > 0].count()[0], 2)) / (phase_nzero + epsilon)
    temp = []
    temp.append(phase_mean)
    result['phase_mean'] = pd.DataFrame(temp)

    # feature 6: phase_sigma
    phase_sigma = 0.0
    for column in train_data.columns:
        phase_sigma += math.pow(train_data[train_data[column] > 0].count()[0] - phase_mean, 2) / (phase_nzero + epsilon)
    phase_sigma = math.sqrt(phase_sigma)
    temp = []
    temp.append(phase_sigma)
    result['phase_sigma'] = pd.DataFrame(temp)

    # feature 7: phase_skew
    phase_skew = 0.0
    for column in train_data.columns:
        phase_skew += math.pow(train_data[train_data[column] > 0].count()[0] - phase_mean, 3) / (phase_nzero + epsilon)
    phase_skew = phase_skew / math.pow(phase_sigma, 3)
    temp = []
    temp.append(phase_skew)
    result['phase_skew'] = pd.DataFrame(temp)

    result['cycle_num'] = cycle_num
    # result = pd.concat((result, mean_data), axis=1)
    return result


train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train1.csv')

train_data = extract_feature(train_data)

file_count = 0
# for i in range(2, 10171):
#     filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train' + str(i) + '.csv'
#     if os.path.exists(filePath):
#         file_count += 1
#         print file_count
#         temp = pd.read_csv(filePath)
#         temp = extract_feature(temp)
#         train_data = pd.concat((train_data, temp), axis=0)

for i in range(1, 10):
    for j in range(1, 10161):
        filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_' + str(i) + '/prps_train' + str(
            j) + '.csv'
        if os.path.exists(filePath):
            file_count += 1
            print file_count
            temp = pd.read_csv(filePath)
            temp = extract_feature(temp)
            train_data = pd.concat((train_data, temp), axis=0)

train_data = train_data.reset_index()
train_data.drop(['index'], axis=1, inplace=True)
train_data.to_csv('E:/JinXiejie/data/PRPS/train_all_feature-7.csv', index=False)
