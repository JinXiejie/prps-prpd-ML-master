import pandas as pd
import os
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import seaborn


def extract_feature(train_data):
    feature = [x for x in train_data.columns if
               x not in ['type_id', 'alarm_id', 'version_id', 'devnum_id', 'channelnum_id', 'devstatus_id', 'datastatus_id', 'type_id',
                         'alarm_id']]
    result = train_data[['type_id', 'alarm_id']].groupby(['type_id']).agg('mean').reset_index()
    cycle_num = train_data.count()[0]
    result['cycle_num'] = cycle_num

    data = train_data[feature]

    for column in data.columns:
        discrete_data = []
        for index, row in data.iterrows():
            discrete_data.append(row[column])
        y = fft(discrete_data)
        y_real = y.real
        y_imag = y.imag

        yf = abs(fft(discrete_data))
        yf1 = abs(fft(discrete_data)) / cycle_num
        yf2 = yf1[range(int(cycle_num / 2))]

        xf = np.arange(len(discrete_data))
        xf1 = xf
        xf2 = xf[range(int(cycle_num / 2))]

        plt.subplot(221)
        plt.plot(row[0:50], discrete_data[0:50])
        plt.title('Original wave')

        plt.subplot(222)
        plt.plot(xf, yf, 'r')
        plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=7, color='#7A378B')

        plt.subplot(223)
        plt.plot(xf1, yf1, 'g')
        plt.title('FFT of Mixed wave(normalization)', fontsize=9, color='r')

        plt.subplot(224)
        plt.plot(xf2, yf2, 'b')
        plt.title('FFT of Mixed wave)', fontsize=10, color='#F08080')

        plt.show()
    m = []

    return result


train_data = pd.read_csv('E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train1.csv')

train_data = extract_feature(train_data)

file_count = 0
for i in range(2, 10171):
    filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_1/prps_train' + str(i) + '.csv'
    if os.path.exists(filePath):
        file_count += 1
        print file_count
        temp = pd.read_csv(filePath)
        temp = extract_feature(temp)
        train_data = pd.concat((train_data, temp), axis=0)

# for i in range(1, 10):
#     for j in range(1, 10161):
#         filePath = 'E:/JinXiejie/data/PRPS/PDMSystemPdmSys_CouplerSPDC-Channel_2_' + str(i) + '/prps_train' + str(
#             j) + '.csv'
#         if os.path.exists(filePath):
#             temp = pd.read_csv(filePath)
#             temp = extract_feature(temp)
#             train_data = pd.concat((train_data, temp), axis=0)

train_data = train_data.reset_index()
train_data.drop(['index'], axis=1, inplace=True)
train_data.to_csv('E:/JinXiejie/data/PRPS/train_feature-7.csv', index=False)
