import os
import pandas as pd
correct = 0
result_nums = 0
same_nums = 0
all_nums = 0
normal_nums = 0
for i in range(1, 1500):

    filePath = 'E:/JinXiejie/PythonCases/PyDemo/ResultData/result' + str(i) + '.csv'
    if os.path.exists(filePath):
        result_nums += 1
        result = pd.read_csv(filePath)
        # get the nums of rows in for every column
        nums = result.count(axis=0)
        all_nums += result.count(axis=0)[5]

        result = result[result['type_id'] == result['label']]
        same_nums += result.count(axis=0)[5]

        half_rate = float(result.count(axis=0)[5]) / float(nums[5])
        if half_rate >= 0.5:
            correct += 1
print result_nums
print correct
print all_nums
print same_nums
print normal_nums


