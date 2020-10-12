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
# tf.nn.relu(a) 输入非线性激活函数relu


# concat 将数据合并
a = tf.ones([4, 35, 8])
b = tf.ones([4, 35, 8])
c = tf.contact([a, b], axis=0)  # TensorShape([8, 35, 8])
d = tf.Stack([a, b], axis=0).shape  # TensorShape([2, 4, 35, 8])
aa, bb = tf.unstack(c, axis=0)  # 解除合并 查询用法
res = tf.split(c, axis=3, num_or_size_splits=[2, 2, 4]  # 把第三个轴按 2 2 4分开

tf.norm(a, ord=1, axis=1)  # 在1维度求向量的1范数
# reduce_min/max/mean  argmax/argmin  tf.equal返回True False

#accuracy
pred = tf.cast(tf.argmax(a, axis=1), dtype=tf.int32)
correct = tf.reduce_sum(tf.equal(y, pred), dtype=tf.int32)
correct = correct/N  # 输出的即为准确度

#张量排序
a = tf.random.shuffle(tf.range(5))   # 打乱序号输出
tf.sort(a, direction='DESCENDING')   # 降序输出
tf.argsort(a, direction='DESCENDING')  # 返回的是排序元素的位置
res = tf.math.top_k(a, 2)  # 取头两个最大的数（概率）
print(res.indices)
print(res.values)

# pad() 函数， tile(a, [1, 2]) axis=0 复制一次  axis=1 复制两次
# broadcast_to(aa, [2, 3, 3])

# clip_by_value()即为通常意义上的限幅
# maximum(a, 2) minimum(a, 8)  这两个其实就是两个判断函数
# clip_by_norm(a, 15)  限制传入向量的范数，即可以限制向量的模的大小
# Gradient clipping: Gradient Exploding or vanishing
# new_grads, total_norm = tf.clip_by_global_norm(grads, 25)  对梯度的模整体都进行缩放，即防止梯度爆炸或者梯度消失