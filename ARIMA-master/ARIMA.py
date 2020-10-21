import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.api import qqplot
import numpy as np

filename = './arima_data.xls'

data = pd.read_excel(filename, index_col=u'时间')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# data.plot()
# plt.title('Time Series')
# plt.show()
# plot_acf(data)
# plt.show()
# plot_pacf(data)
# plt.show()
# print(u'原始序列的ADF检验结果为：', ADF(data[u'风速']))
D_data = data.diff(periods=1).dropna()
D_data.columns = [u'一阶差分']
# print(D_data)
# D_data.plot()
# plt.show()
# plot_acf(D_data)
# plt.show()
# plot_pacf(D_data)
# plt.show()
# print(u'1阶差分序列的ADF检验结果为：', ADF(D_data[u'一阶差分']))
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))
from statsmodels.tsa.arima_model import ARIMA
data[u'风速'] = data[u'风速'].astype(float)
pmax = int(len(D_data)/10)
qmax = int(len(D_data)/10)

# bic_matrix = []
# for p in range(4):
#     tmp = []
#     for q in range(4):
#         try:
#             tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
#         except:
#             tmp.append(None)
#     bic_matrix.append(tmp)
# bic_matrix = pd.DataFrame(bic_matrix)
# print("bic矩阵为：", end="")
# print(bic_matrix)
# p, q = bic_matrix.stack().idxmin()
# print(u'bic最小的P值和q值为：%s、%s' %(p, q))

model = ARIMA(data, (4, 1, 3)).fit()
# model.summary2()
num = 40
forecast, fcasterr, conf_int = model.forecast(num)

print("预测序列为：", end="")
print(forecast)
print(conf_int)

data = np.array(data)
data = data.tolist()
# print(data)

dd = []
for i in data:
    for j in i:
        dd.append(j)
# print(dd)
min = []
max = []
ss = []

for i in conf_int:
    for j in i:
        ss.append(j)

for i in range(0, 2*num-1, 2):
    min.append(ss[i])

for i in range(1, 2*num+1, 2):
    max.append(ss[i])
# print(min)
# print(max)
qidian = 191
plt.plot(range(1, 191, 1), dd, label='yuanlai')
plt.plot(range(qidian, qidian+num, 1), forecast, label='yuce')
plt.fill_between(range(qidian, qidian+num, 1), min, max, facecolor='grey', alpha=0.3)
plt.show()

resid = model.resid
# plt.plot(resid)
# plt.show()

fig = plt.figure(figsize=(20, 20))
ax4 = plt.subplot(221)
fig = plt.plot(resid)
ax1 = plt.subplot(222)
fig = plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = plt.subplot(223)
fig = plot_pacf(resid, lags=40, ax=ax2)
ax = plt.subplot(224)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()
print(sm.stats.durbin_watson(model.resid.values))

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111)
# fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()





