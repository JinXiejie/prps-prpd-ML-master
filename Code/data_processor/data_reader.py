import pandas as pd
import os
file_path = 'G:\特高频GIS局放数据\prps_raw_data'
train_data = pd.DataFrame()
for i in range(1, 42313):
    path = file_path + '\prps_train_' + str(i) + '.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        train_data = pd.concat([train_data, df], ignore_index=True)
        print("处理文件：" + str(i))
# save_path = '/Code/data/prps_data.csv'
save_path = 'G:\PycharmProjects\prps-prpd-ML-master\Code\data\prps_data.csv'
train_data.to_csv(save_path, index=False)