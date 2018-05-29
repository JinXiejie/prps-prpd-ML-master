import numpy as np
import pandas as pd
from scipy.misc import imsave

prps_data = pd.read_csv('E:\JinXiejie\data\PRPS\prps_train11.csv')
imsave('E:\JinXiejie\data\PRPS\prps_train11.jpg', prps_data)

x = np.random.random((60, 80, 3))
imsave('E:\JinXiejie\data\PRPS\meelo.jpg', x)
