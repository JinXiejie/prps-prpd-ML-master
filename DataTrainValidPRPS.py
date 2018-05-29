import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import operator
from matplotlib import pylab as plt

# number of train: 37700;
# number of type_id in 0: 540;
# number of type_id in 2: 20219;
# number of type_id in 5: 1185;
# number of type_id in 7: 397;
# number of type_id in 10: 15359;
train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_all_data.csv')

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train_data['type_label'] = type_in_train

num_class = train_data['type_label'].max() + 1
print len(train_data)
print len(train_data[train_data['type_id'] == 0])
print len(train_data[train_data['type_id'] == 2])
print len(train_data[train_data['type_id'] == 5])
print len(train_data[train_data['type_id'] == 7])
print len(train_data[train_data['type_id'] == 10])
# num_trees = 450
params = {
    'objective': 'multi:softmax',
    'eta': 0.01,
    'max_depth': 9,
    # 'subsample': 0.7,
    'lambda': 10,
    'eval_metric': 'merror',
    'seed': 1024,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

feature = [x for x in train_data.columns if x not in ['type_id', 'type_label']]
X_train, X_test, y_train, y_test = train_test_split(train_data[feature], train_data['type_label'].astype('int'),
                                                    test_size=0.2, random_state=0)
xgbtrain = xgb.DMatrix(X_train, y_train)
xgbeval = xgb.DMatrix(X_test, y_test)
watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)

xgbtrain = xgb.DMatrix(train_data[feature], train_data['type_label'])
watchlist = [(xgbtrain, 'train')]
cv_log = xgb.cv(params, xgbtrain, num_boost_round=5000, nfold=5)
# model = xgb.train(params, xgbtrain, num_boost_round=7000, early_stopping_rounds=20, evals=watchlist)

bst_merror = cv_log['test-merror-mean'].min()
cv_log['nb'] = cv_log.index
cv_log.index = cv_log['test-merror-mean']
bst_nb = cv_log.nb.to_dict()[bst_merror]

cv_log['bst_nb'] = bst_nb
cv_log['bst_merror'] = bst_merror
cv_log.to_csv('data/cv_log-feature-6.csv', index=None)


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
importance = model.get_score(fmap='xgb.fmap', importance_type='gain')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'score'])
df.to_csv('data/feature_importance-feature-6.csv', index=None)
df['score'] = df['score'] / df['score'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='score', legend=False, figsize=(16, 30))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('data/feature_importance_xgb-6.png')
