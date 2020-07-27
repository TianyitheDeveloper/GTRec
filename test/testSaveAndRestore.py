"""
保存模型参数，再从保存的模型中恢复

2020年7月15日 17点35分
"""
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

# 需要在会话中保存
with tf.Session() as sess:
    v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")       # 定义变量v1形状
    v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")       # 定义变量v2形状
    init_op = tf.global_variables_initializer()                 # 初始化v1和v2的值
    saver = tf.train.Saver()                                    # 定义保存器
    sess.run(init_op)                                           # 运行初始化操作
    print("v1:", sess.run(v1))                                  # 运行初始化v1操作
    print("v2:", sess.run(v2))                                  # 运行初始化v2操作
    saver_path = saver.save(sess, "save/model.ckpt")            # 保存会话
    print("Model saved in file:", saver_path)

# 需要在会话中恢复
with tf.Session() as sess:
    weight_saver = tf.train.import_meta_graph('save/model.ckpt.meta')   # 从保存的图中引入参数
    weight_saver.restore(sess,tf.train.latest_checkpoint('save/'))      # 恢复
    graph = tf.get_default_graph()                                      # 获取默认图
    v1_1 = graph.get_tensor_by_name("v1:0")                             # 获取v1值
    v2_1 = graph.get_tensor_by_name("v2:0")                             # 获取v2值
    print(sess.run(v1_1))
    print(sess.run(v2_1))