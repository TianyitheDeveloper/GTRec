"""
tensorflow的卷积层(实验只做卷积，不做训练)

tf.nn.conv2d (input,
              filter,
              strides,
              padding,
              use_cudnn_on_gpu=None,
              data_format=None,
              name=None)

input：
输入的要做卷积的图片，要求为一个张量，shape为[batch, in_height, in_weight, in_channel]，其中batch为图片的数量，in_height为图片高度，
in_weight为图片宽度，in_channel 为图片的通道数。
filter：
卷积核，要求也是一个张量，shape为 [filter_height, filter_weight, in_channel, out_channels]，其中filter_height为卷积核高度，
filter_weight为卷积核宽度，in_channel是图像通道数 ，和input的in_channel要保持一致，out_channel是卷积核数量。
strides：
卷积时在图像每一维的步长，这是一个一维的向量，[1, strides, strides, 1]，第一位和最后一位固定必须是1
padding：
string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

input = tf.Variable(tf.random_normal([2,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))
op1 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

input = tf.Variable(tf.random_normal([2,3,3,5]))
filter = tf.Variable(tf.random_normal([2,2,5,1]))
op2 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

input = tf.Variable(tf.random_normal([2,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
op3 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

input = tf.Variable(tf.random_normal([2,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
op4 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='VALID')

input = tf.Variable(tf.random_normal([2,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
op5 = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

input = tf.Variable(tf.random_normal([2,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))
op6 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

input = tf.Variable(tf.random_normal([2,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))
op7 = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("op1:2批3×3的5通道图片，高为1宽为1入通道为5出通道为1的卷积核，考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批3乘3通道为1的图片")
    print(sess.run(op1))
    print("op2:2批3×3的5通道图片，高为2宽为2入通道为5出通道为1的卷积核，考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批3乘3通道为1的图片")
    print(sess.run(op2))
    print("op3:2批3×3的5通道图片，高为3宽为3入通道为5出通道为1的卷积核，不考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批1乘1通道为1的图片")
    print(sess.run(op3))
    print("op4:2批5×5的5通道图片，高为3宽为3入通道为5出通道为1的卷积核，不考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批3乘3通道为1的图片")
    print(sess.run(op4))
    print("op5:2批5×5的5通道图片，高为3宽为3入通道为5出通道为1的卷积核，考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批5乘5通道为1的图片")
    print(sess.run(op5))
    print("op6:2批5×5的5通道图片，高为3宽为3入通道为5出通道为7的卷积核，考虑边界，"
          "固定步长为[1,1,1,1]，输出为2批5乘5通道为7的图片")
    print(sess.run(op6))
    print("op6:2批5×5的5通道图片，高为3宽为3入通道为5出通道为7的卷积核，考虑边界，"
          "固定步长为[1,2,2,1]，输出为2批3乘3通道为7的图片")
    print(sess.run(op7))


