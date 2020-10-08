import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')
warnings.filterwarnings('ignore')


#任务 2.1绘制生鲜类商品和一般商品每天销售金额的折线图，并分析比较两类产品的销售状况。

data = pd.read_csv('./task1_1.csv', encoding='gbk')
data.loc[:, '销售日期'] = pd.to_datetime(data.loc[:, '销售日期'].astype(str), format='%Y-%m-%d', errors='coerce')

data_new = data[['商品类型', '销售日期', '销售金额']]
#print(data_new)

#检查每个列的缺失值的占比
#print(data_new.apply(lambda x: sum(x.isnull())/len(x), axis=0))#注意0和1的区别

data_query1 = data_new.loc[:, '商品类型'] == '生鲜'
data_query2 = data_new.loc[:, '商品类型'] == '一般商品'
#print(data_query1)
data_fresh = data_new[data_query1]
data_comm = data_new[data_query2]
print(data_fresh)# !!!!!这种输出的写法要学会

# 根据销售日期分组，求生鲜类商品和一般商品的每天销售金额
data_fresh_cost = data_fresh.groupby('销售日期').sum()
data_fresh_cost.columns = {'销售金额'}

data_comm_cost = data_comm.groupby('销售日期').sum()
data_comm_cost.columns = {'销售金额'}
print(data_fresh_cost)

plt.style.use('ggplot')

# 设置图框的大小
fig = plt.figure(figsize=(10, 6))
# 前两个1表示共有1*1个子图，最后一个1表示第1个子图
ax = fig.add_subplot(1, 2, 1)
bx = fig.add_subplot(1, 2, 2)
plt.subplot(121)
# 绘图--生鲜类商品销售折线图
plt.plot(data_fresh_cost.index,  # x轴数据
         data_fresh_cost.values,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='blue',  # 折线颜色
         marker='o',  # 点的形状
         markersize=6,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='blue',
         label='生鲜类商品')  # 点的填充色

# 添加标题和坐标轴标签
plt.title('每日销售金额折线图', fontproperties=font)
plt.xlabel('日期', fontproperties=font)
plt.ylabel('销售金额', fontproperties=font)
# 显示label
plt.legend(prop=font)

# 日期刻度标签60度倾斜
fig.autofmt_xdate(rotation=90)

plt.subplot(122)
# 绘图--一般商品销售折线图
plt.plot(data_comm_cost.index,  # x轴数据
         data_comm_cost.values,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='#ff9999',  # 折线颜色
         marker='o',  # 点的形状
         markersize=6,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='#ff9999',
         label='一般商品')  # 点的填充色

# 添加标题和坐标轴标签
plt.title('每日销售金额折线图', fontproperties=font)
plt.xlabel('日期', fontproperties=font)
plt.ylabel('销售金额', fontproperties=font)
# 显示label
plt.legend(prop=font)

# 日期刻度标签60度倾斜
fig.autofmt_xdate(rotation=90)

# 设置x轴显示密度间隔为5
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
bx.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))

# 显示图形
plt.show()

# 配合收藏网页看
data_new = data[['大类名称', '销售日期', '销售金额']]
data_new['月份'] = [x.month for x in data_new['销售日期']]


#print(data_new['大类名称'].value_counts(dropna=False))
data_month1 = data_new.loc[data_new['月份'] == 1, :]
data_month2 = data_new.loc[data_new['月份'] == 2, :]
data_month3 = data_new.loc[data_new['月份'] == 3, :]
data_month4 = data_new.loc[data_new['月份'] == 4, :]

# 根据大类名称分组，求各大类商品的每月销售金额
data_month_cost1 = data_month1.groupby('大类名称').sum()[['销售金额']]
print(data_month_cost1)   # 有没有后缀是明显不一样的
data_month_cost2 = data_month2.groupby('大类名称').sum()[['销售金额']]
data_month_cost3 = data_month3.groupby('大类名称').sum()[['销售金额']]
data_month_cost4 = data_month4.groupby('大类名称').sum()[['销售金额']]

# 画饼图

# 设置图框的大小
fig = plt.figure(figsize=(10, 8))
# 前两个1表示共有1*1个子图，最后一个1表示第1个子图
ax = fig.add_subplot(1, 1, 1)

s = np.array(data_month_cost1)
b = []
for i in s:
    for ii in i:
        ii = int(ii)
        b.append(ii)

print(b)

# 绘制饼图，textprops={'fontproperties':font}显示中文

plt.pie(x=b,
        labels=data_month_cost1.index,
        autopct='%.1f%%',
        shadow=False,
        startangle=90,
        center=(3, 3),
        textprops={'fontproperties': font})

# 添加标题，fontproperties=font显示中文
plt.title("1月份销售金额", fontproperties=font)
# 显示图例，prop=font显示中文
plt.legend(prop=font)
# 饼图保持圆形
plt.axis('equal')
# 显示图像
plt.show()

