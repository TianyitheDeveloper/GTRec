import tensorflow as tf
import os
from tensorflow.keras import layers
import os
# 选择编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 创建模型
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))
# 设置目标函数和学习率
optimizer = tf.keras.optimizers.SGD(0.2)
# 编译模型
model.compile(loss='binary_crossentropy', optimizer=optimizer)
# 输出模型概况
model.summary()














tf.print("success")
