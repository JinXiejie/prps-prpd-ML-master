import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_feature-7.csv')

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train_data['type_label'] = type_in_train

num_class = train_data['type_label'].max() + 1
params = {
    'objective': 'multi:softmax',
    'eta': 0.01,
    'max_depth': 9,
    'lambda': 20,
    'min_child_weight ': 10,
    'scale_pos_weight': 10,
    # 'gamma': 0.01,
    'eval_metric': 'merror',
    'seed': 0,
    'missing': -999,
    'num_class': num_class,
    'silent': 1,
}

# feature = [x for x in train.columns if x not in ['type_id', 'version_id', 'devnum_id', 'channel_id',
#                                                           'devstatus_id', 'datastatus_id', 'type_label']]
feature = [x for x in train_data.columns if x not in ['type_id', 'type_label', 'alarm_id']]

X_train, X_test, y_train, y_test = train_test_split(train_data[feature], train_data['type_label'].astype('int'),
                                                    test_size=0.2, random_state=0)

# train = train_data.reset_index()
# for idx in range(0, 5):
#     train = train[train['index'] % 5 != idx]
#     test = train[train['index'] % 5 == idx]
#     stacking_feature = [x for x in train.columns if x not in ['type_id', 'type_label', 'alarm_id']]
#     xgbtrain = xgb.DMatrix(train[stacking_feature], train['type_label'])
#     watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
#     model = xgb.train(params, xgbtrain, num_boost_round=500, early_stopping_rounds=50, evals=watchlist)
#     y_pred = pd.DataFrame(test['index'])
#     y_pred['xgb_'] = rfreg.predict(test[stacking_feature])
#     rf_pred = rf_pred.append(y_pred)


# xgbtrain = xgb.DMatrix(X_train, y_train)
# xgbeval = xgb.DMatrix(X_test, y_test)
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)
# 0.876908

xgbtrain = xgb.DMatrix(X_train, y_train)
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=240, early_stopping_rounds=20, evals=watchlist)
y_pred = pd.DataFrame(y_test).reset_index()
y_pred.drop(['index'], axis=1, inplace=True)
xgbtest = xgb.DMatrix(X_test)
y_pred['xgb_pred'] = pd.DataFrame(model.predict(xgbtest))


def error(predict, y_test):
    n = len(predict)
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    print predict
    print y_test
    for row_index in range(n):
        if int(predict[row_index]) == y_test[row_index]:
            TP += 1
        if int(predict[row_index]) != 0 and y_test[row_index] == 0:
            FN += 1
        if int(predict[row_index]) == 0 and y_test[row_index] != 0:
            FP += 1
        if int(predict[row_index]) != y_test[row_index]:
            TN += 1
    print "TP: " + str(TP)
    print "FN: " + str(FN)
    print "FP: " + str(FP)
    print "TN: " + str(TN)
    print "准确率: " + str(float(TP + TN)/float(TP + TN + FP + FN))
    print "精确率: " + str(float(TP)/float(TP + FP))
    print "召回率: " + str(float(TP)/float(TP + FN))
    return float(TP) / float(n)
error(y_pred['xgb_pred'], y_test)

# rf_scaler = StandardScaler()
# rf_data = pd.c
# rf_scaler.fit(X_train)
# X_train_rf = rf_scaler.transform(X_train)
# X_test_rf = rf_scaler.transform(X_test)

random_forest_clf = RandomForestClassifier(n_estimators=500, max_depth=15)
random_forest_clf.fit(X_train, y_train)
y_pred['rf_pred'] = pd.DataFrame(random_forest_clf.predict(X_test))
# 0.872476612506


mlp_scaler = StandardScaler()
mlp_data = pd.concat((X_train, X_test))
mlp_scaler.fit(mlp_data)
X_train_mlp = mlp_scaler.transform(X_train)
X_test_mlp = mlp_scaler.transform(X_test)

mlp_clf = MLPClassifier(hidden_layer_sizes=(13, 13, 13, 13), max_iter=500)
mlp_clf.fit(X_train_mlp, y_train)
y_pred['nn_pred'] = pd.DataFrame(mlp_clf.predict(X_test_mlp))
# 0.881339241753
y_pred.astype(int)


def vote_for_pred(x1, x2, x3):
    arr = [x1, x2, x3]
    # count the arr[i] happens in the arr by dict method
    res = {k: arr.count(k) for k in set(arr)}
    count = 0
    shop_id = None
    for (k, v) in res.items():
        if count < v:
            shop_id = k
            count = v
    if count >= 2:
        return shop_id
    else:
        return x3


y_pred_vote = list(
    map(lambda x1, x2, x3: vote_for_pred(x1, x2, x3), y_pred['xgb_pred'], y_pred['rf_pred'], y_pred['nn_pred']))


def error_clf(predict, y_test):
    n = len(predict)
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    count = 0
    for row_index in range(n):
        if int(predict[row_index]) == y_test[row_index]:
            count += 1
    return float(count) / float(n)


print (error_clf(y_pred['xgb_pred'], y_test))
print (error_clf(y_pred['rf_pred'], y_test))
print (error_clf(y_pred['nn_pred'], y_test))
print (error_clf(y_pred_vote, y_test))
y_pred = pd.concat((y_test, y_pred), axis=1)
temp = pd.DataFrame(y_pred_vote)
temp = pd.concat((temp, y_pred), axis=1)
y_pred['blending'] = y_pred_vote
y_pred.to_csv('E:/JinXiejie/prps-prpd-ML-master/prps-prpd-ML-master/prps_pred_vote.csv', index=None)
