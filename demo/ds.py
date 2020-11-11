import numpy as np
from sklearn.model_selection import train_test_split

dense = np.loadtxt('./new.csv', delimiter=',')
bayes = np.loadtxt('./bayes.csv', delimiter=',')
svm = np.loadtxt('./svm.csv', delimiter=',')
# print(dense)
X, Y = dense.shape
fusion = []

# for i in range(X):
#     for j in range(Y):
#         if dense[i][j] > 0.95:
#             dense[i][j] = 0.9
#         elif dense[i][j] < 0.1:
#             dense[i][j] = 0.1
#
# for i in range(X):
#     for j in range(Y):
#         if bayes[i][j] > 0.95:
#             bayes[i][j] = 0.9
#         elif bayes[i][j] < 0.1:
#             bayes[i][j] = 0.1
#
# for i in range(X):
#     for j in range(Y):
#         if svm[i][j] > 0.95:
#             svm[i][j] = 0.9
#         elif svm[i][j] < 0.1:
#             svm[i][j] = 0.1

print(bayes)

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


def fuse(a, b):
    K = a[0] * (b[1] + b[2]) + a[1] * (b[0] + b[2]) + a[2] * (b[0] + b[1])
    m0 = (a[0] * b[0])/(1 - K)
    m1 = (a[1] * b[1]) / (1 - K)
    m2 = (a[2] * b[2]) / (1 - K)
    return [m0, m1, m2]


def maxnum(a):
    num = 0
    if a[1] > num:
        num = 1
    if a[2] > num:
        num = 2
    return num


for i in range(X):
    dense_pro = dense[i]
    bayes_pro = bayes[i]
    svm_pro = svm[i]
    pro12 = fuse(dense_pro, bayes_pro)
    pro123 = fuse(pro12, svm_pro)
    print(pro123)
    num = maxnum(pro123)
    fusion.append(num)

show_accuracy(np.array(fusion), y_train, '训练集')
print(fusion)
