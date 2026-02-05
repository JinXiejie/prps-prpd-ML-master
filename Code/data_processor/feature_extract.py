import pandas as pd
import os, glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
import math

def category_processor(cat_data, embedding_dim=8):
    label_encoder = LabelEncoder()
    ids_vector = pd.DataFrame()
    for column in cat_data.columns:
        print('processing column_name: ', column)
        id_fea_val = cat_data[column]
        print(id_fea_val.value_counts())
        data_nums = len(id_fea_val.value_counts())
        id_fea_val = label_encoder.fit_transform(id_fea_val)
        embeddings_layer = nn.Embedding(num_embeddings=data_nums, embedding_dim=embedding_dim)
        embeddings_vector = embeddings_layer(torch.tensor(id_fea_val))
        vector = embeddings_vector.detach().numpy()
        col_name = []
        for i in range(embedding_dim):
            col_name.append(column + '_' + str(i))
        vector = pd.DataFrame(vector, columns=col_name)
        ids_vector = pd.concat([ids_vector, vector], axis=1)
    return ids_vector

def extract_feature(train_data):
    # train_data = pd.read_csv('G:\特高频GIS局放数据\prps_raw_data\prps_train_1.csv')
    # del_feature = ['version_id', 'addr', 'devstatus_id', 'probSt', 'coeff', 'phaseNum', 'avg', 'avgV', 'fiftyHZ',
    #                'hundredHZ']
    # feature = [x for x in train_data.columns if x not in del_feature]
    # train_data = train_data[feature]


    ids_feature = ['type_id', 'datetime', 'devnum_id', 'channelnum_id', 'datastatus_id', 'alarm_id', 'cycNum', 'intens']

    train_data.loc[:,'datetime'] = train_data['datetime'].apply(lambda x: str(x).split(' ')[0])
    train_data.loc[:,'intens'] = train_data['intens'].apply(lambda x: x * math.pow(10, 45))
    result = train_data[ids_feature].groupby(['type_id', 'datetime', 'devnum_id', 'channelnum_id'
                       , 'datastatus_id', 'alarm_id', 'cycNum', 'intens']).count().reset_index()

    cycle_num = len(train_data)
    phase_feature = [x for x in feature if x not in ids_feature]
    phase_data = train_data[phase_feature]
    mean_data = train_data.groupby(['type_id'])[phase_feature].agg('mean').reset_index()
    result = pd.merge(result, mean_data, how='left', on='type_id')

    column_nums = phase_data.count(axis=1)[0]
    column_nums_half = int(column_nums / 2)
    epsilon = 0.0001

    posi_phase = []
    for i in range(column_nums_half):
        posi_phase.append('p' + str(i))

    neg_phase = []
    for i in range(column_nums_half, column_nums):
        neg_phase.append('p' + str(i))
    posi_phase_feature = [x for x in phase_data.columns if x in posi_phase]
    neg_phase_feature = [x for x in phase_data.columns if x in neg_phase]

    # feature 1: posi_neg_q
    # feature of balance between positive q in positive phase and negative q in negative phase
    posi_num = 0
    posi_sum = 0
    neg_num = 0
    neg_sum = 0
    for index, row in phase_data.iterrows():
        for column in phase_data[posi_phase_feature].columns:
            if row[column] > 0:
                posi_num += 1
                posi_sum += row[column]
        for column in phase_data[neg_phase_feature].columns:
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
    for index, row in phase_data.iterrows():
        for column in phase_data[posi_phase_feature].columns:
            if row[column] > 0:
                first_posi = int(column.split('p')[1]) + 1
                break
        for column in phase_data[neg_phase_feature].columns:
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

    for column in phase_data[posi_phase_feature].columns:
        nzero_column = len(phase_data[phase_data[column] > 0])
        if float(nzero_column) / float(cycle_num) > 0.1:
            posi_phase_width += 1
    for column in phase_data[neg_phase_feature].columns:
        nzero_column = len(phase_data[phase_data[column] > 0])
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
    for column in phase_data.columns:
        phase_nzero += len(phase_data[phase_data[column] > 0])
    for column in phase_data.columns:
        phase_mean += float(math.pow(len(phase_data[phase_data[column] > 0]), 2)) / (phase_nzero + epsilon)
    temp = []
    temp.append(phase_mean)
    result['phase_mean'] = pd.DataFrame(temp)

    # feature 6: phase_sigma
    phase_sigma = 0.0
    for column in phase_data.columns:
        phase_sigma += math.pow(len(phase_data[phase_data[column] > 0]) - phase_mean, 2) / (phase_nzero + epsilon)
    phase_sigma = math.sqrt(phase_sigma)
    temp = []
    temp.append(phase_sigma)
    result['phase_sigma'] = pd.DataFrame(temp)

    # feature 7: phase_skew
    phase_skew = 0.0
    for column in phase_data.columns:
        phase_skew += math.pow(len(phase_data[phase_data[column] > 0]) - phase_mean, 3) / (phase_nzero + epsilon)
    if phase_sigma == 0:
        phase_skew = 0
    else:
        phase_skew = phase_skew / math.pow(phase_sigma, 3)
    temp = []
    temp.append(phase_skew)
    result['phase_skew'] = pd.DataFrame(temp)

    return result
# result = pd.concat((result, mean_data), axis=1)



print("读取数据中：。。。")

train_data = pd.read_csv('G:\特高频GIS局放数据\prps_raw_data\prps_train_1.csv')
del_feature = ['version_id', 'addr', 'devstatus_id', 'probSt', 'coeff', 'phaseNum', 'avg', 'avgV', 'fiftyHZ', 'hundredHZ']
feature = [x for x in train_data.columns if x not in del_feature]
train_data = extract_feature(train_data[feature])

file_count = 0
# for i in range(2, 100):
for i in range(1, 42171):
    filePath = 'G:\特高频GIS局放数据\prps_raw_data\prps_train_' + str(i) + '.csv'
    if os.path.exists(filePath):
        file_count += 1
        print(file_count)
        temp = pd.read_csv(filePath)
        temp = extract_feature(temp[feature])
        train_data = pd.concat((train_data, temp), axis=0)

train_data.to_csv('G:\PycharmProjects\prps-prpd-ML-master\Code\data\prps_fea_data.csv', index=False)
