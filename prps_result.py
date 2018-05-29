import os
import pandas as pd

correct = 0
result_nums = 0
same_nums = 0
all_nums = 0
normal_nums = 0

filePath = 'E:/JinXiejie/PythonCases/PyDemo/ResultData/result.csv'
if os.path.exists(filePath):
    result_nums += 1
    result = pd.read_csv(filePath)
    # get the nums of rows in for every column
    nums = result.count(axis=0)
    result = result[result['type_id'] > 2]
    temp = result.count()[0]
    x = temp**2
    all_nums += result.count(axis=0)[0]

    result = result[result['type_id'] == result['label']]
    same_nums += result.count(axis=0)[0]

print (all_nums)
print (same_nums)
print (str(float(same_nums) / float(all_nums) * 100.0) + "%")


