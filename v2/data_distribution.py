# coding:utf-8
import seaborn as sns
import matplotlib.pyplot as plt  # 约定俗成的写法plt
import numpy as np
import os
from scipy.stats import norm
import pandas as pd

rs = np.random.RandomState(50)  # 设置随机数种子
# randn()的是形状，下面代码 = N(0, 10000)，即期望为0，标准差为100的正态分布
# s = pd.Series(rs.randn(100) * 10)
# x_y = [[1, 2], [3, 4], [5, 6], [7, 8]]
x_y = []
input_path = "on-ascii-train/"

print("开始读取点")
for parent, dirnames, filenames in os.walk(input_path, followlinks=True):
    for filename in filenames:
        file_path = os.path.join(parent, filename)
        print('文件名：%s' % filename)
        print('文件完整路径：%s\n' % file_path)
        if file_path[-6:] == ".ascii":
            temp_open = open(file_path, "r")
            while 1:
                lines = temp_open.readlines(100000)
                if not lines:
                    break
                for line in lines:
                    temp_list = line.split()
                    x_y.append([float(temp_list[0]), float(temp_list[1])])

print("点读取完毕")

plt.figure(figsize=(8, 4))
# sns.distplot(x, hist=True, kde=False, norm_hist=False, rug=True, vertical=False, label='distplot',
#              axlabel='feature', hist_kws={'color': 'y', 'edgecolor': 'k'}, fit=norm)
# sns.distplot(y, bins=len(y), hist=True, kde=False, norm_hist=False, rug=True, vertical=False, label='distplot',
#              axlabel='feature', hist_kws={'color': 'y', 'edgecolor': 'k'}, fit=norm)
#
# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(np.array(x_y), columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)

# # 用标准正态分布拟合
plt.legend()
plt.grid(linestyle='--')
plt.savefig('my_data_distribution.png')
plt.show()
