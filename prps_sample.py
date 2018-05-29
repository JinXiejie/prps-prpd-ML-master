import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/train_all_data.csv')

mainLabelEncoder = LabelEncoder()
type_label = train_data[['type_id']]
mainLabelEncoder.fit(type_label)

type_in_train = train_data[['type_id']]
type_in_train = mainLabelEncoder.transform(type_in_train)

train_data['type_label'] = type_in_train

# df0 = train_data[train_data['type_id'] == 0]  # 148
# for i in range(100):
#     temp = train_data[train_data['type_id'] == 0].apply(lambda t: t.sample(int(len(t) * 0.2), axis=0, random_state=1))
#     df0 = pd.concat((df0, temp))
# df2 = train_data[train_data['type_id'] == 2]  # 6315
# df5 = train_data[train_data['type_id'] == 5]  # 166
# for i in range(100):
#     temp = train_data[train_data['type_id'] == 5].apply(lambda t: t.sample(int(len(t) * 0.2), axis=0, random_state=1))
#     df5 = pd.concat((df5, temp))
# df7 = train_data[train_data['type_id'] == 7]  # 80
# for i in range(200):
#     temp = train_data[train_data['type_id'] == 7].apply(lambda t: t.sample(int(len(t) * 0.2), axis=0, random_state=1))
#     df7 = pd.concat((df7, temp))
# df10 = train_data[train_data['type_id'] == 10]  # 3451
# print len(df0)
# print len(df2)
# print len(df5)
# print len(df7)
# print len(df10)

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


# xgbtrain = xgb.DMatrix(X_train, y_train)
# xgbeval = xgb.DMatrix(X_test, y_test)
# watchlist = [(xgbtrain, 'train'), (xgbeval, 'evaluate')]
# model = xgb.train(params, xgbtrain, num_boost_round=2000, early_stopping_rounds=50, evals=watchlist)
# 0.876908

xgbtrain = xgb.DMatrix(X_train, y_train)
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'evaluate')]
model = xgb.train(params, xgbtrain, num_boost_round=750, early_stopping_rounds=20, evals=watchlist)
y_pred = pd.DataFrame(y_test).reset_index()
y_pred.drop(['index'], axis=1, inplace=True)
xgbtest = xgb.DMatrix(X_test)
y_pred['xgb_pred'] = pd.DataFrame(model.predict(xgbtest))


def error_clf(predict, y_test):
    n = len(predict)
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    count = 0
    for row_index in range(n):
        if int(predict[row_index]) == y_test[row_index]:
            count += 1
    return float(count) / float(n)


def norm_error_clf(predict, y_test):
    n = len(predict)
    print len(y_test[y_test['type_label'] == 0])
    predict = (np.array(predict)).tolist()
    y_test = (np.array(y_test)).tolist()
    count = 0
    for row_index in range(n):
        if int(predict[row_index][0]) == 0 and y_test[row_index][0] != 0:
            print row_index
            count += 1
    print count
    return float(count) / float(n)

print norm_error_clf(y_pred[['xgb_pred']], y_pred[['type_label']])
# 107
# 1
# 0.000132625994695



