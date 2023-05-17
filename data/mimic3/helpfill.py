import pandas as pd

# 读取p1.csv文件
df = pd.read_csv('p4.csv')

# 使用上一行的值填充缺失值
df.fillna(method='ffill', inplace=True)

# 保存修改后的数据到p1.csv
df.to_csv('p4.csv', index=False)

