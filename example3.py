import tensorflow as tf
from tensorflow import keras

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x.shape, y.shape, x.min(), x.max(), x.mean())
print(x_test.shape, y_test.shape)
print(y[:2])
y_onehot = tf.one_hot(y, depth=10)
print(y_onehot[:2])

(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(x.shape, y.shape, x.min(), x.max(), x.mean())
db = tf.data.Dataset.from_tensor_slices(x_test)
print(next(iter(db)).shape)  # 迭代读取数据

db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# db = db.shuffle(10000)  # 将数据打乱顺序


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


db2 = db.map(preprocess)  # 对每一个sample都做这样的预处理
res = next(iter(db2))
print(res[0].shape, res[1].shape)

db3 = db2.batch(32)


