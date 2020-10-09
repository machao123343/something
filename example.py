import tensorflow as tf
import numpy as np

print(tf.constant(1))
print(tf.constant(1.))
# print(tf.constant(2.2, dtype=tf.int32))  # 不能强制进行类型转换
print(tf.constant(2., dtype=tf.double))
print(tf.constant([True, False]))

print(tf.constant('hello world'))
a = tf.constant(1)
b = tf.constant(tf.ones([1, 2, 3]))
print(tf.rank(a))
print(tf.rank(b))
print(np.arange(1, 4))

a = tf.constant(1)
b = tf.constant(1.)
c = tf.constant([True, False])
d = np.arange(1, 4)
isinstance(a, tf.Tensor)
tf.is_tensor(b)
print(a.dtype, b.dtype, c.dtype)
print(a.dtype == tf.float32)

a = np.arange(5)
aa = tf.convert_to_tensor(a)
aa = tf.convert_to_tensor(a, dtype=tf.int32)
print(aa)
aaa = tf.cast(aa, dtype=tf.float32)
print(aaa)

b = tf.constant([0, 1])
bb = tf.cast(b, dtype=tf.bool)
print(bb)
tf.cast(bb, tf.int32)
print(tf.cast(bb, tf.int32))

