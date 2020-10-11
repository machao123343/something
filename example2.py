import tensorflow as tf
import numpy as np

c = np.ones((2, 2))
print(c)
tf.convert_to_tensor(c)

# tf.convert_to_tensor(np.zeros(2, 3))
# tf.convert_to_tensor([1, 2])  # [1, 2.]  [1], [2.]   编写形势要按照上面的格式

# tf.zeros()的用法

a = tf.range(10)
print(a[-1])  # 9
print(a[-1:])  # [9]
print(a[-2:])  # (8, 9)
print(a[:-1])  # (0 1 2 3 4 5 6 7 8)
print(a[:-2])  # (0 1 2 3 4 5 6 7)  切片slice   indexby start:end:step
# [:14]从最开始到14  [14:]从14到结尾    ::-1逆序采集

a = tf.range(4)
print(a[2::-2])  #从第2（3）个元素开始逆向采集
#  a[..., 0]前面的全部取，最后取0索引

#  [classes, students, subjects]
tf.gather(a, axis=0, indices=[2, 3]).shape  #这个gather的作用记一下，用到现查也来得及
tf.gather_nd(a, [0, 1]).shape  # gather_nd()也是上面的情况

a = tf.random.normal([4, 28, 28, 3])

tf.reshape(a, [4, 784, 3]).shape     # 对a进行维度变换

tf.rashape(a, [4, -1, 3]).shape  # -1是根据总维度自动计算填补的，所以不能有两个-1
tf.transpose(a, perm=[0, 1, 3, 2]).shape  # 相当于维度位置变换，即数据位置的旋转

# Expand dim
tf.expand_dims(a, axis=0).shape  # 在相应的坐标位置扩展维度，可以查询

tf.squeeze(tf.zeros[1, 2, 1, 1, 3]).shape
tf.squeeze(tf.zeros[1, 2, 1, 1, 3], axis=0).shape  # squeeze删除维度为1的轴

# broadcasting  对张量的维度再进行扩张
(x + tf.random.normal([4, 1, 1, 1])).shape
# TensorShape([4, 32, 32, 3])
b = tf.broadcast_to(tf.random.normal([4, 1, 1, 1], [4, 32, 32, 3])).shape

# element-wise  matrix-wise dim-wise
# tf.exp() tf.math.log() pow(b, 3) == b ** 3
# a@b == tf.matmul(a, b)
#tf.nn.relu(a) 输入非线性激活函数relu

