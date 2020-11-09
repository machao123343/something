from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

x_data = load_iris().data  # 返回 iris 数据集所有输入特征
y_data = load_iris().target  # 返回 iris 数据集所有标签


def get_accuary(x_test, y_test, w, b):
    """
    计算准确率的函数
    x_test    : 测试数据集
    y_label   : 标签(以独热码的形式传入)
    w         : 参数w
    b         : 参数b
    """
    # 预测标签
    y_pred = tf.matmul(x_test, w) + b
    # 将预测值转化为概率分布(在此处此步可以省略)
    y_pred = tf.nn.softmax(y_pred, axis=1)
    # # 求预测值中的最大值
    y_pred = tf.argmax(y_pred, axis=1)
    # 求标签中的最大值
    y_label = tf.argmax(y_test, axis=1)
    # 计算准确率
    acc = tf.reduce_mean(tf.cast(y_pred == y_label, tf.float64))
    # 返回准准确率
    return acc.numpy()


x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.3,
                                                    random_state=2718)


y_train = tf.one_hot(y_train, 3, dtype=tf.float64)  # 转化为独热码
y_test = tf.one_hot(y_test, 3, dtype=tf.float64)

# 配成[输入特征，标签]对
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

# 初始化模型参数
w = tf.Variable(tf.random.truncated_normal([4, 3],
                                           stddev=0.1,
                                           dtype=tf.float64))
b = tf.Variable(tf.random.truncated_normal([3],
                                           stddev=0.1,
                                           dtype=tf.float64))
# 设置学习率
lr = 0.01
# 设置训练轮数
epoch = 1000
for i in range(epoch):  # 训练轮数层面迭代开始
    for step, (x_batch, y_batch) in enumerate(train_db):  # batch 层面迭代开始
        with tf.GradientTape() as tape:
            # 前向传播计算 y, y为预测结果
            y = tf.matmul(x_train, w) + b
            y = tf.nn.softmax(y)
            # 计算损失函数
            loss = (y - y_train) ** 2
            loss = tf.reduce_mean(loss, axis=0)

        # 求导
        grads = tape.gradient(loss, [w, b])
        # 更新参数
        w.assign_sub(lr * grads[0])
        b.assign_sub(lr * grads[1])
    # 打印训练结果
    if i % 50 == 0:
        print("Epoch {}, loss: {}".format(i + 1, loss))
        # 计算训练集上的准确率
        # acc = get_accuary(x_train, y_train, w, b)
        acc = get_accuary(x_test, y_test, w, b)
        print("Accuracy {}".format(acc))


print(y)
np.savetxt('new.csv', y, delimiter=',')


