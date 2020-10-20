import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.api import qqplot

filename='./arima_data.xls'
forrecastnum=5
data=pd.read_excel(filename,index_col=u'时间')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data.plot()
plt.title('Time Series')
plt.show()
plot_acf(data)
plt.show()
plot_pacf(data)
plt.show()
print(u'原始序列的ADF检验结果为：',ADF(data[u'风速']))
D_data=data.diff(periods=1).dropna()
D_data.columns=[u'一阶差分']
print(D_data)
D_data.plot()
plt.show()
plot_acf(D_data)
plt.show()
plot_pacf(D_data)
plt.show()
print(u'1阶差分序列的ADF检验结果为：',ADF(D_data[u'一阶差分']))
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：',acorr_ljungbox(D_data,lags=1))
from statsmodels.tsa.arima_model import ARIMA
data[u'风速'] = data[u'风速'].astype(float)
pmax=int(len(D_data)/10)
qmax=int(len(D_data)/10)
bic_matrix=[]
for p in range(4):
    tmp=[]
    for q in range(4):
        try:
            tmp.append(ARIMA(data,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)
print("bic矩阵为：", end="")
print(bic_matrix)
p,q=bic_matrix.stack().idxmin()
print(u'bic最小的P值和q值为：%s、%s'%(p,q))
model=ARIMA(data,(3,1,3)).fit()
# model.summary2()
forecast=model.forecast(5)
print("预测序列为：", end="")
print(forecast)
resid = model.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(resid, lags=40, ax=ax2)
plt.show()
print(sm.stats.durbin_watson(model.resid.values))

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()





