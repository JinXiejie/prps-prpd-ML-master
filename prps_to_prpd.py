import pandas as pd
import os
import numpy as np


def prpd_transform(prps_data):
    type_id = prps_data['type_id'][0]
    feature = [x for x in prps_data.columns if
               x not in ['version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                         'alarm_id']]
    prps_data = prps_data[feature]
    # phase_columns = prps_data.columns
    max_value = np.mat(prps_data).max()
    prps_data = pd.DataFrame(np.matrix(prps_data[feature]).transpose())
    l = []
    print ('------------------- prpd_dict -------------------')
    rows = str(len(prps_data))
    for index, row in prps_data.iterrows():
        print ('------------------- row: ' + str(row) + '(' + rows + ')' + ' -------------------')
        prpd_dict = {}
        for column in prps_data.columns:
            # if row[column] not in prpd_dict:
            #     prpd_dict[row[column]] = 1
            # else:
            #     prpd_dict[row[column]] += 1
            if row[column] != 0:
                if row[column] not in prpd_dict:
                    prpd_dict[row[column]] = 1
                else:
                    prpd_dict[row[column]] += 1
        l.append(prpd_dict)

    prpd_array = []
    print ('------------------- prpd_array -------------------')
    length = str(len(l))
    count = 0
    for row in l:
        count += 1
        print ('------------------- row: ' + str(count) + '(' + length + ')' + ' -------------------')
        prpd_array.append(row)
    print ('------------------- over -------------------')
    prpd_array = pd.DataFrame(prpd_array)

    prpd_array = pd.DataFrame(prpd_array, columns=range(0, int(max_value) + 1))
    prpd_array = prpd_array.fillna(0)
    prpd_data = pd.DataFrame(np.matrix(prpd_array).transpose())
    # prpd_data = pd.DataFrame(prpd_data, columns=phase_columns)
    prpd_data['type_id'] = type_id
    return prpd_data

prps_data = pd.read_csv('D:/PRPS_Corona1.csv')
orginal_prpd_data = pd.read_csv('D:/PRPD_Corona1.csv')
prps_data['type_id'] = 1
prpd_data = prpd_transform(prps_data)

for i in range(1, 10180):
    PRPSfilePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train' + str(i) + '.csv'
    if os.path.exists(PRPSfilePath):
        prpd_data = prpd_transform(pd.read_csv(PRPSfilePath))
        PRPDfilePath = 'E:/JinXiejie/data/PRPD/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prpd_train' + str(i) + '.csv'
        prpd_data.to_csv(PRPDfilePath, index=None)

