import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

result0 = pd.read_csv('ResultData/result_binary.csv')
result = pd.read_csv('ResultData/result_binary0.csv')
# result1 = pd.read_csv('ResultData/result_binary2.csv')
# result2 = pd.read_csv('ResultData/result_binary5.csv')
# result3 = pd.read_csv('ResultData/result_binary7.csv')
# result4 = pd.read_csv('ResultData/result_binary10.csv')
result = result[['index', 'type_id', 'type_label', 'result_score']]
for i in range(1, 11):
    result_path = 'ResultData/result_binary' + str(i) + '.csv'
    if os.path.exists(result_path):
        sub_result = pd.read_csv(result_path)
        result = pd.merge(result, sub_result[['index', 'type_id', 'type_label', 'result_score']], how='left',
                          on='index')
result.to_csv('ResultData/result_binary.csv', index=None)
