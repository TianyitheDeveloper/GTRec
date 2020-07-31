"""
tensorflow的增维操作

TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数。
当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，但是有些时候在构建图的过程
中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary
or unicode string, got 1 在这种情况下，我们就可以考虑使用expand_dims来将维度加1。比
如我自己代码中遇到的情况，在对图像维度降到二维做特定操作后，要还原成四维[batch, height,
width, channels]，前后各增加一维。如果用reshape，则因为上述原因报错
"""

import tensorflow as tf
a = [
    [
        [1,1,1,1],
        [2,2,2,2],
        [3,3,3,3],
        [4,4,4,4],
        [5,5,5,5,]
    ],
    [
        [6,6,6,6],
        [7,7,7,7],
        [8,8,8,8],
        [9,9,9,9],
        [0,0,0,0]
    ],
]

# shape会发生很大的变化
print(tf.constant(a))
print(tf.expand_dims(a,0))
print(tf.expand_dims(a,-1))
