from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split

gnb = GaussianNB()


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '正确率：', np.mean(acc))


path = 'iris.data'  # 数据文件路径)
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
x, y = np.split(data, (4,), axis=1)
# x = x[:, :2]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
# print(x)
# print(y)
gnb.fit(x_train, y_train.ravel())
print(gnb.score(x_train, y_train))  # 精度
y_pred = gnb.predict(x_train)
Y_pred = gnb.predict_proba(x_train)
print('预测softmax:', Y_pred)
print(Y_pred.shape)
show_accuracy(y_pred, y_train, '训练集')
print(gnb.score(x_test, y_test))
y_hat = gnb.predict(x_test)
show_accuracy(y_hat, y_test, '测试集')

np.savetxt('bayes.csv', Y_pred, delimiter=',')

