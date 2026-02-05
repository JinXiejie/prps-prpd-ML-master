import pandas as pd
import os, glob, math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import operator
from matplotlib import pylab as plt
import torch
import torch.nn as nn
import seaborn as sns
from datetime import datetime


# number of train: 37700;
# number of type_id in 0: 650;
# number of type_id in 2: 21823;
# number of type_id in 5: 1457;
# number of type_id in 7: 583;
# number of type_id in 10: 17521;

# judge if str can be date format
def is_valid_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def cal_date_diff(date_str):
    today = datetime.strptime('2014-6-11', '%Y-%m-%d')
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return abs(today - date).days


def category_processor(cat_data, embedding_dim=8):
    label_encoder = LabelEncoder()
    ids_vector = pd.DataFrame()
    for column in cat_data.columns:
        print('processing column_name: ', column)
        id_fea_val = cat_data[column]
        # print(id_fea_val.value_counts())
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


data = pd.read_csv('G:\PycharmProjects\prps-prpd-ML-master\Code\data\prps_fea_data.csv')
print(data['cycNum'].value_counts())
data['cycNum'].value_counts().to_csv('G:\PycharmProjects\prps-prpd-ML-master\Code\data\cyc_num.csv', index=True)
# process label
mainLabelEncoder = LabelEncoder()
type_label = mainLabelEncoder.fit_transform(data[['type_id']])
data['type_label'] = type_label
num_class = data['type_label'].max() + 1

# process id feature
ids_feature = ['devnum_id', 'channelnum_id', 'datastatus_id', 'alarm_id']
ids_data = category_processor(data[ids_feature])
ids_feature.append('type_id')
data = data[[x for x in data.columns if x not in ids_feature]]
data = pd.concat([data, ids_data], axis=1)

# train and test data split
# datetime = pd.DataFrame(data['datetime']).value_counts()

data['datetime'] = data['datetime'].apply(lambda x: cal_date_diff(str(x)) if is_valid_date(str(x)) else -1)
# datetime = pd.DataFrame(data['datetime']).value_counts()
# datetime.to_csv('G:\PycharmProjects\prps-prpd-ML-master\Code\data\datetime.csv', index=True)

# 575:'2015-12-31'
# 600:'2016-2-1'
train_data = data[data['datetime'] < 575]
valid_data = data[(575 <= data['datetime']) & (data['datetime'] < 600)]
test_data = data[data['datetime'] >= 600]
# train_data = data['datetime'].apply(lambda x: x < '2016-1-1')
# valid_data = data['datetime'].apply(lambda x: '2016-1-1' <= x < '2016-2-1')
# test_data = data['datetime'].apply(lambda x: x >= '2016-2-1')

feature = [x for x in data.columns if x not in ['type_label']]
x_train = train_data[feature]
y_train = train_data['type_label']
x_eval = valid_data[feature]
y_eval = valid_data['type_label']
x_test = test_data[feature]
y_test = test_data['type_label']

# sns.displot(train_data.datetime.values)
# plt.show()


# num_trees = 450
params = {
    'objective': 'multi:softmax',
    'eta': 0.01,
    'max_depth': 9,
    'subsample': 0.9,
    'lambda': 100,
    'eval_metric': 'merror',
    'seed': 1024,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

xgb_train = xgb.DMatrix(x_train, y_train)
xgb_eval = xgb.DMatrix(x_eval, y_eval)
watchlist = [(xgb_train, 'train'), (xgb_eval, 'evaluate')]

model = xgb.train(params, xgb_train, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)
# [311]	train-merror:0.154651	evaluate-merror:0.175197
# Stopping. Best iteration:
# [261]	train-merror:0.160556	evaluate-merror:0.172244
x_train = pd.concat([x_train, x_eval])
y_train = pd.concat([y_train, y_eval])
xgb_train = xgb.DMatrix(x_train, y_train)
watchlist = [(xgb_train, 'train'), (xgb_train, 'evaluate')]
model = xgb.train(params, xgb_train, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)

y_pred = pd.DataFrame(y_test)
xgb_test = xgb.DMatrix(x_test)
y_pred['pred'] = model.predict(xgb_test)


def norm_error_clf(predict, y_test):
    samples = len(predict)
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    count = 0
    for row_index in range(samples):
        # if predict[row_index][0] < 0.5 and y_test[row_index][0] != 0:
        if predict[row_index][0] == y_test[row_index][0]:
            count += 1
    print("count:" + str(count))
    print("samples:" + str(samples))
    return float(count) / float(samples)


print("准确率:" + str(norm_error_clf(y_pred[['pred']], y_pred[['type_label']])))


# count:3546
# samples:4257
# 准确率:0.8329809725158562

# count:3618
# samples:4257
# 准确率:0.8498942917547568

# [1996]	train-merror:0.05781	evaluate-merror:0.05781
# [1997]	train-merror:0.05781	evaluate-merror:0.05781
# [1998]	train-merror:0.05776	evaluate-merror:0.05776
# [1999]	train-merror:0.05776	evaluate-merror:0.05776
# count:3680
# samples:4257
# 准确率:0.8644585388771435

# [5563]	train-merror:0.00614	evaluate-merror:0.00614
# [5564]	train-merror:0.00614	evaluate-merror:0.00614
# [5565]	train-merror:0.00614	evaluate-merror:0.00614
# count:3689
# samples:4257
# 准确率:0.8665727037820061

#
# # plot the map of feature importance
# def ceate_feature_map(features):
#     outfile = open('xgb.fmap', 'w')
#     i = 0
#     for feat in features:
#         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
#         i = i + 1
#
#     outfile.close()
#
#
# model = xgb.train(params, xgb_train, num_boost_round=2000, evals=watchlist)
# ceate_feature_map(feature)
#
# # importance = model.get_fscore(fmap='xgb.fmap')
# importance = model.get_score(fmap='xgb.fmap')
# importance = sorted(importance.items(), key=operator.itemgetter(1))
#
# df = pd.DataFrame(importance, columns=['feature', 'score'])
# df.to_csv('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/data/feature_importance-feature_vector.csv', index=None)
# df['score'] = df['score'] / df['score'].sum()
#
# plt.figure()
# df.plot()
# df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(16, 30))
# plt.title('XGBoost Feature Importance')
# plt.xlabel('relative importance')
# plt.gcf().savefig('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/data/feature_importance_xgb-vector.png')
