import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

# train = pd.read_csv('E:/JinXiejie/data/PRPS/train_feature-7.csv')
train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_all_data.csv')
# number of train: 37700;
# number of type_id in 0: 540;
# number of type_id in 2: 20219;
# number of type_id in 5: 1185;
# number of type_id in 7: 397;
# number of type_id in 10: 15359;
df1 = train_data[train_data['type_id'] == 0]
df2 = train_data[train_data['type_id'] != 0]
df1['type_label'] = 0
df2['type_label'] = 1

params = {
    'objective': 'binary:logistic',
    'eta': 0.01,
    'max_depth': 9,
    'subsample': 0.8,
    'lambda': 5,
    'eval_metric': 'auc',
    'seed': 1024,
    'missing': -999,
    'silent': 1,
}
train_data = pd.concat((df1, df2))
feature = [x for x in train_data.columns if x not in ['type_id', 'alarm_id', 'type_label']]
X_train, X_test, y_train, y_test = train_test_split(train_data[feature], train_data['type_label'].astype('int'),
                                                    test_size=0.2, random_state=0)

for idx in range(5):
    XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.95, random_state=idx)

    print len(XX_train)
    xgbtrain = xgb.DMatrix(XX_train, yy_train)
    xgbeval = xgb.DMatrix(XX_test, yy_test)
    watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
    model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)

    # xgbtrain = xgb.DMatrix(X_train, y_train)
    # watchlist = [(xgbtrain, 'train')]
    # model = xgb.train(params, xgbtrain, num_boost_round=900, evals=watchlist)
# 2:10; n = 1600; 1783, 1954, 0.912487205732
# 0:2 830
y_pred = pd.DataFrame(y_test)
xgbtest = xgb.DMatrix(X_test)
y_pred['pred'] = model.predict(xgbtest)


def error_clf(predict, y_test):
    n = len(predict)
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    # print predict
    # print y_test
    count = 0
    for row_index in range(n):
        # print y_test[row_index]
        # print predict[row_index]
        if (y_test[row_index][0] == 1 and predict[row_index][0] >= 0.5) or (
                        y_test[row_index][0] == 0 and predict[row_index][0] < 0.5):
            count += 1
    print count
    print n
    return float(count) / float(n)


def norm_error_clf(predict, y_test):
    n = len(predict)
    print len(y_test[y_test['type_label'] == 0])
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    count = 0
    for row_index in range(n):
        # if predict[row_index][0] < 0.5 and y_test[row_index][0] != 0:
        if predict[row_index][0] < 0.5:
            print row_index
            count += 1
    print count
    return float(count) / float(n)


print norm_error_clf(y_pred[['pred']], y_pred[['type_label']])
