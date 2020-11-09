import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('iris.data', header=None)
print(data)
antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864']

data_setosa0 = data[data[4] == 'Iris-setosa'][0]
data_versicolor0 = data[data[4] == 'Iris-versicolor'][0]
data_virginica0 = data[data[4] == 'Iris-virginica'][0]
data_SepalLengthCm = pd.DataFrame({'setosa': data_setosa0, 'versicolor': data_versicolor0, 'virginica': data_virginica0})  # 自动对齐行数

# print(data_SepalLengthCm)

data_setosa1 = data[data[4] == 'Iris-setosa'][1]
data_versicolor1 = data[data[4] == 'Iris-versicolor'][1]
data_virginica1 = data[data[4] == 'Iris-virginica'][1]
data_SepalWidthCm = pd.DataFrame({'setosa': data_setosa1, 'versicolor': data_versicolor1, 'virginica': data_virginica1})

data_setosa2 = data[data[4] == 'Iris-setosa'][2]
data_versicolor2 = data[data[4] == 'Iris-versicolor'][2]
data_virginica2 = data[data[4] == 'Iris-virginica'][2]
data_PetalLengthCm = pd.DataFrame({'setosa': data_setosa2, 'versicolor': data_versicolor2, 'virginica': data_virginica2})

data_setosa3 = data[data[4] == 'Iris-setosa'][3]
data_versicolor3 = data[data[4] == 'Iris-versicolor'][3]
data_virginica3 = data[data[4] == 'Iris-virginica'][3]
data_PetalWidthCm = pd.DataFrame({'setosa': data_setosa3, 'versicolor': data_versicolor3, 'virginica': data_virginica3})

f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
sns.despine(left=True)
sns.violinplot(data=data_SepalLengthCm, linewidth=1, width=0.8, palette=antV, ax=axes[0, 0])
axes[0, 0].set_xlabel('Kinds')
axes[0, 0].set_ylabel('SepalLengthCm')
sns.violinplot(data=data_SepalWidthCm, linewidth=1, width=0.8, palette=antV, ax=axes[0, 1])
axes[0, 1].set_xlabel('Kinds')
axes[0, 1].set_ylabel('SepalWidthCm')
sns.violinplot(data=data_PetalLengthCm, linewidth=1, width=0.8, palette=antV, ax=axes[1, 0])
axes[1, 0].set_xlabel('Kinds')
axes[1, 0].set_ylabel('PetalLengthCm')
sns.violinplot(data=data_PetalWidthCm, linewidth=1, width=0.8, palette=antV, ax=axes[1, 1])
axes[1, 1].set_xlabel('Kinds')
axes[1, 1].set_ylabel('PetalWidthCm')
plt.show()

g = sns.pairplot(data=data, palette=antV, hue=4, kind='scatter')
'''
kind：用于控制非对角线上的图的类型，可选"scatter"与"reg"
将 kind 参数设置为 "reg" 会为非对角线上的散点图拟合出一条回归直线，更直观地显示变量之间的关系。
diag_kind：控制对角线上的图的类型，可选"hist"与"kde"
hue ：针对某一字段进行分类
palette：控制色调
'''
# data.rename(columns={"sepal_length": "萼片长",
#                      "sepal_width": "萼片宽",
#                      "petal_length": "花瓣长",
#                      "petal_width": "花瓣宽",
#                      "species": "种类"}, inplace=True)
# kind_dict = {
#     "setosa": "山鸢尾",
#     "versicolor": "杂色鸢尾",
#     "virginica": "维吉尼亚鸢尾"
# }
# data["种类"] = data["种类"].map(kind_dict)
# # data.head()  # 数据集的内容如下
plt.show()

plt.subplots(figsize=(10, 8))
pd.plotting.andrews_curves(data, 4, color=['blue', 'black', 'yellow'])
plt.show()
