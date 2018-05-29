import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.parser import parse
from datetime import date
from sklearn import preprocessing
import xgboost as xgb
from enum import Enum

train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/train.csv')
test_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_3/test.csv')

feature = [x for x in train_data.columns if x not in ['type_id']]

lda = LinearDiscriminantAnalysis(n_components=4)
lda.fit(train_data[feature], train_data['type_id'])
train_data_lda = lda.transform(train_data[feature])
lda.fit(test_data[feature], test_data['type_id'])
test_data_lda = lda.transform(test_data[feature])

train_data_lda = pd.DataFrame(train_data_lda)
test_data_lda = pd.DataFrame(test_data_lda)

train_data_lda = pd.concat((train_data['type_id'], train_data_lda), axis=1)

test_data_lda = pd.concat((test_data['type_id'], test_data_lda), axis=1)
train_data_lda.plot(kind="scatter", x="0", y="1")
