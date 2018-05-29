import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
import math


def extract_feature(train_data):
    feature = [x for x in train_data.columns if
               x not in ['type_id', 'alarm_id', 'version_id', 'devnum_id', 'channelnum_id', 'devstatus_id',
                         'datastatus_id', 'type_id', 'alarm_id']]
    result = train_data[['type_id', 'alarm_id']].groupby(['type_id']).agg('mean').reset_index()
    cycle_num = train_data.count()[0]
    result['cycle_num'] = cycle_num

    data = train_data[feature]

    for column in data.columns:
        phase_vector = ""
        for index, row in data.iterrows():
            phase_vector += str(row[column]) + ' '
        result[column] = phase_vector

    feature = [x for x in train_data.columns if
               x not in ['version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                         'alarm_id']]
    mean_data = train_data.groupby(['type_id'])[feature].agg('mean').reset_index()
    train_data = train_data[feature]
    column_nums = train_data.count(axis=1)[0]
    column_nums_half = column_nums / 2
    epsilon = 0.0001

    result = pd.merge(result, mean_data, how='right', on='type_id')
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
for i in range(2, 10171):
    # for i in range(2, 10171):
    filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train' + str(i) + '.csv'
    if os.path.exists(filePath):
        file_count += 1
        print file_count
        temp = pd.read_csv(filePath)
        temp = extract_feature(temp)
        train_data = pd.concat((train_data, temp), axis=0)
# excute here!
train_data.to_csv('E:/JinXiejie/data/PRPS/train_feature_vector.csv', index=False)

train_data = train_data.reset_index()
train_data.drop(['index'], axis=1, inplace=True)

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train_data['type_label'] = type_in_train

num_class = train_data['type_label'].max() + 1

# train_x = train_data[['alarm_id', 'cycle_num']]
# vector_feature = [x for x in train_data.columns if x not in ['type_id', 'alarm_id', 'cycle_num', 'type_label']]
vector_feature = []
for i in range(64):
    s = 'p' + str(i) + '_x'
    vector_feature.append(s)

cv = CountVectorizer(min_df=1)
data = train_data[vector_feature]

num_feature = vector_feature
num_feature.append('type_id')
num_feature.append('type_label')
num_feature = [x for x in train_data.columns if x not in num_feature]
train_x = train_data[num_feature]

# train_x = sparse.hstack(train_x)
vector_feature.remove('type_id')
vector_feature.remove('type_label')
for feature in vector_feature:
    cv.fit(train_data[feature])
    train_a = cv.fit_transform(data[feature])
    train_x = sparse.hstack((train_x, train_a))
    # print(cv.get_feature_names())
    # print(train_a.toarray())
print('cv prepared !')

# train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_feature_vector.csv')
# df = pd.read_csv('E:/JinXiejie/data/PRPS/train_feature-7.csv')
# feature = [x for x in df.columns if x not in ['type_id', 'cycle_num', 'alarm_id']]
# df = df[feature]
# temp = pd.merge(df, train_data, how='left', on=['type_id', 'cycle_num', 'alarm_id'])
# temp = pd.concat((df, train_data), axis=1)
# print len(train_data)
# print len(df)

# num_trees = 450
params = {
    'objective': 'multi:softmax',
    'eta': 0.01,
    'max_depth': 9,
    'subsample': 0.7,
    'lambda': 100,
    'eval_metric': 'merror',
    'seed': 1024,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

X_train, X_test, y_train, y_test = train_test_split(train_x, train_data['type_label'].astype('int'),
                                                    test_size=0.2, random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]

model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)
# [311]	train-merror:0.154651	evaluate-merror:0.175197
# Stopping. Best iteration:
# [261]	train-merror:0.160556	evaluate-merror:0.172244


xgbtrain = xgb.DMatrix(train_x, train_data['type_label'])
watchlist = [(xgbtrain, 'train')]
cv_log = xgb.cv(params, xgbtrain, num_boost_round=5000, early_stopping_rounds=50, nfold=5)
# model = xgb.train(params, xgbtrain, num_boost_round=7000, early_stopping_rounds=20, evals=watchlist)

bst_merror = cv_log['test-merror-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-merror-mean']
bst_nb = cv_log.nb.to_dict()[bst_merror]

cv_log['bst_nb'] = bst_nb
cv_log['bst_merror'] = bst_merror
cv_log.to_csv('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/data/cv_log-feature_vector.csv', index=None)


# plot the map of feature importance
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


model = xgb.train(params, xgbtrain, num_boost_round=bst_nb + 50, evals=watchlist)
ceate_feature_map(feature)

# importance = model.get_fscore(fmap='xgb.fmap')
importance = model.get_score(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'score'])
df.to_csv('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/data/feature_importance-feature_vector.csv', index=None)
df['score'] = df['score'] / df['score'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(16, 30))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/data/feature_importance_xgb-vector.png')
