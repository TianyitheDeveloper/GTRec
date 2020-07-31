"""
tensorflow中的转置操作
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

# 直接输出a
print(tf.constant(a))
# 按照原张量逆序转置
print(tf.transpose(a))
# 实际上没有转置
print(tf.transpose(a,[0,1,2]))
# 只转置后两个维度
print(tf.transpose(a,[0,2,1]))