import tensorflow as tf
from tensorflow import keras

# (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
# print(x.shape, y.shape, x.min(), x.max(), x.mean())
# print(x_test.shape, y_test.shape)
# print(y[:2])
# y_onehot = tf.one_hot(y, depth=10)
# print(y_onehot[:2])
#
# (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
# print(x.shape, y.shape, x.min(), x.max(), x.mean())
# db = tf.data.Dataset.from_tensor_slices(x_test)
# print(next(iter(db)).shape)  # 迭代读取数据
#
# db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# # db = db.shuffle(10000)  # 将数据打乱顺序
#
#
# def preprocess(x, y):
#     x = tf.cast(x, dtype=tf.float32)/255
#     y = tf.cast(y, dtype=tf.int32)
#     y = tf.one_hot(y, depth=10)
#     return x, y
#
#
# db2 = db.map(preprocess)  # 对每一个sample都做这样的预处理
# res = next(iter(db2))
# print(res[0].shape, res[1].shape)
#
# db3 = db2.batch(32)
#
# db_iter = iter(db3)
# while True:
#     next(db_iter)   会报错
#
# db4 = db3.repeat(2)  # 在for遍历的时候会遍历两次


def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.0
    y = tf.cast(y, dtype=tf.int64)
    return x, y


def mnist_dataset():
    (x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()
    y = tf.one_hot(y, depth=10)
    y_val = tf.one_hot(y_val, depth=10)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)  # 打散再取100batch
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    ds_val = ds_val.map(prepare_mnist_features_and_labels)
    ds_val = ds_val.shuffle(10000).batch(100)
    return ds, ds_val


