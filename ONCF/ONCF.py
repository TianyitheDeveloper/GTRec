import math
import os
import numpy as np
import tensorflow.compat.v1 as tf
from time import time
import argparse
import LoadData as DATA
import logging
tf.compat.v1.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Run ONCF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='输入数据路径')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='排序列表数量')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='数据集')
    parser.add_argument('--epoch', type=int, default=300,
                        help='运行轮次数')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='预训练标记. 1: 预训练; 0: 随机初始化; -1: 保存模型到预训练')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批量大小')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='隐向量维度')
    parser.add_argument('--lamda', type=float, default=0,
                        help='二元线性正则化项')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout保留率（1: 不dropout）')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='损失类型 (square_loss或 log_loss)')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='优化器类型 (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='每隔verbose次显示结果')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='是否批正则化 (0 or 1)')
    parser.add_argument('--net_channel', nargs='?', default='[32,32,32,32,32,32]',
                        help='网络通道（6层）')
    parser.add_argument('--regs', nargs='?', default='[0,0,0]',
                        help='正则化项（用户、物品嵌入，全连接层权重，CNN卷积核权重）')
    return parser.parse_args()


class ONCF():
    def __init__(self,
                 user_field_M,
                 item_field_M,
                 pretrain_flag,
                 save_file,
                 hidden_factor,
                 epoch,
                 batch_size,
                 learning_rate,
                 lamda_bilinear,
                 keep,
                 optimizer_type,
                 batch_norm,
                 verbose,
                 random_seed=2020):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.lambda_weight = regs[2]

        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # 输入数据.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # 变量
            self.weights = self._initialize_weights()   # 初始化嵌入与对应权重
            self.nc = eval(args.net_channel)            # 网络通道
            iszs = [1]+self.nc[:-1]                      # 输入
            oszs = self.nc                              # 输出
            self.P = []                                 # 卷积核
            self.P.append(self._conv_weight(iszs[0],oszs[0]))
            for i in range(1,6):
                self.P.append(self._conv_weight(iszs[0],oszs[0]))   # 前五层
            self.W = self.weight_variable([self.nc[-1],1])          # 最后一层
            self.b = self.weight_variable([1])

            # 模型
            # 正样本
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],
                                                                  self.user_features)
            self.user_embedding = tf.reduce_sum(self.user_feature_embeddings, 1, keep_dims=True)
            self.positive_item_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                   self.positive_features)
            self.positive_embedding = tf.reduce_sum(self.positive_item_embeddings, 1, keep_dims=True)
            self.relation = tf.matmul(tf.transpose(self.user_embedding, perm=[0, 2, 1]), self.positive_embedding)
            self.net_input = tf.expand_dims(self.relation, -1)
            self.layer = []
            positive_input = self.net_input
            for p in self.P:
                self.layer.append(
                    self._conv_layer(positive_input, p))
                positive_input = self.layer[-1]
            self.dropout_positive = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_positive = tf.matmul(tf.reshape(self.dropout_positive, [-1, self.nc[-1]]), self.W) + self.b
            self.user_feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features),
                1)  # None * 1
            self.item_feature_bias_positive = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features),
                1)  # None * 1
            self.positive = tf.add_n(
                [self.interaction_positive, self.user_feature_bias, self.item_feature_bias_positive])
            # 负样本
            self.negative_item_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                   self.negative_features)
            self.negative_embedding = tf.reduce_sum(self.negative_item_embeddings, 1, keep_dims=True)
            self.relation = tf.matmul(tf.transpose(self.user_embedding, perm=[0, 2, 1]), self.negative_embedding)
            self.net_input = tf.expand_dims(self.relation, -1)
            self.layer = []
            negative_input = self.net_input
            for p in self.P:
                self.layer.append(
                    self._conv_layer(negative_input, p, ))
                negative_input = self.layer[-1]
            self.dropout_negative = tf.nn.dropout(self.layer[-1], self.dropout_keep)
            self.interaction_negative = tf.matmul(tf.reshape(self.dropout_negative, [-1, self.nc[-1]]), self.W) + self.b
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features),
                1)  # None * 1
            self.negative = tf.add_n(
                [self.interaction_negative, self.user_feature_bias, self.item_feature_bias_negative])  # None * 1

            # 计算损失
            self.loss = -tf.log(tf.sigmoid(self.positive-self.negative))
            self.loss = tf.reduce_sum(self.loss)
            self.loss = self.loss + \
                        self.lambda_bilinear * \
                        (tf.reduce_sum(tf.square(self.user_embedding)) +
                         tf.reduce_sum(tf.square(self.positive_embedding)) +
                         tf.reduce_sum(tf.square(self.negative_embedding))) + \
                        self.gamma_bilinear * \
                        self._regular([(self.W, self.b)]) + \
                        self.lambda_weight * \
                        (self._regular(self.P) + self._regular([(self.W, self.b)]))

            # 优化器
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate,initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # 初始化
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        """
        初始化嵌入
        :return:
        """
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
            print("load!")
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
        return all_weights

    def _conv_weight(self, isz, osz):
        """
        初始化卷积核
        :param isz: 输入大小
        :param osz: 输出大小
        :return:
        """
        return (self.weight_variable([2,2,isz,osz]),self.bias_variable([osz]))

    def weight_variable(self, shape):
        """
        根据形状初始化权重
        :param param: 形状
        :return: tf变量型的初始化权重
        """
        # 初始化shape形状的参数，产生截断正态分布随机数，取值范围为[ mean - 2 * stddev, mean + 2 * stddev ]
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """
        根据形状初始化偏置
        :param param: 形状
        :return: tf变量型的初始化偏置
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _conv_layer(self, input, P):
        """
        设置二维卷积层
        :param input: 输入张量
        :param P: 卷积核
        :return: 二维卷积层
        """
        conv = tf.nn.conv2d(input,P[0],strides=[1,2,2,1],padding='VALID')
        return tf.nn.relu(conv+P[1])

    def _regular(self, P):
        res = 0
        for p in P:
            res+=tf.reduce_sum(tf.square(p[0]))+tf.reduce_sum(tf.square(p[1]))
        return res

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, train_data, batch_size):  # generate a random block of training data
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items.values()
        # get sample
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            # uniform sampler
            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id = data.binded_users[user_features]  # get userID
            pos = data.user_positive_list[user_id]  # get positive list for the userID
            # candidates = list(set(all_items) - set(pos))  # get negative set
            neg = np.random.randint(len(all_items))  # uniform sample a negative itemID from negative set
            while (neg in pos):
                neg = np.random.randint(len(all_items))
            negative_feature = data.item_map[neg].strip().split('-')  # get negative item feature
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user': X_user, 'X_positive': X_positive, 'X_negative': X_negative}

    def train(self, Train_data):
        for epoch in range(self.epoch):
            total_loss = 0
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            for i in range(total_batch):
                # 生成一个批量
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            logger.info("第%d个轮次的损失是%f" % (epoch, total_loss))
            if (epoch + 1) % 100 == 0:
                model.evaluate()
        print("结束训练，开始保存")
        if self.pretrain_flag < 0:
            print("保存模型参数以作为预训练模型")
            self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0]
        rank = [[], [], [], []]
        topK = [5, 10, 15, 20]
        for index in range(len(data.Test_data['X_user'])):
            user_features = data.Test_data['X_user'][index]
            item_features = data.Test_data['X_item'][index]
            scores = model.get_scores_per_user(user_features)
            # 获取真实物品分数
            true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
            true_item_score = scores[true_item_id]
            # 删除已浏览分数
            user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]
            visited = data.user_positive_list[user_id]
            scores = np.delete(scores, visited)
            # 是否命中
            sorted_scores = sorted(scores, reverse=True)

            label = []
            for i in range(len(topK)):
                label.append(sorted_scores[topK[i] - 1])
                if true_item_score >= label[i]:
                    count[i] = count[i] + 1
                    rank[i].append(sorted_scores.index(true_item_score) + 1)
        for i in range(len(topK)):
            mrr = 0
            ndcg = 0
            hr = float(count[i] / len(data.Test_data['X_user']))
            for item in rank[i]:
                mrr += float(1.0) / item
                ndcg += float(1.0) / np.log2(item + 1)
            mrr /= len(data.Test_data['X_user'])
            ndcg /= len(data.Test_data['X_user'])
            k = (i + 1) * 5
            logger.info("\n前%d个物品的HR,NDCG分别为%f,%f" % (k, hr, ndcg))

    def get_scores_per_user(self, user_feature):  # evaluate the results for an user context, return scorelist
        # num_example = len(Testdata['Y'])
        # get score list for a userID, store in scorelist, indexed by itemID
        scorelist = []

        # X_item = []
        # Y=[[1]]
        all_items = data.binded_items.values()
        # true_item_id=data.binded_items[item]
        # user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],X_user)

        if len(all_items) % self.batch_size == 0:
            batch_count = len(all_items) / self.batch_size
            flag = 0
        else:
            batch_count = math.ceil(len(all_items) / self.batch_size)
            flag = 1
        j = 0
        # print(len(all_items))
        # print(batch_count)
        # print(flag)
        for i in range(int(batch_count)):
            X_user, X_item = [], []
            if flag == 1 and i == batch_count - 1:
                k = len(all_items)
            else:
                k = j + self.batch_size
            # print(j)
            # print(k)
            for itemID in range(j, k):
                X_user.append(user_feature)
                item_feature = [int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
                X_item.append(item_feature)
            feed_dict = {self.user_features: X_user, self.positive_features: X_item, self.train_phase: False,
                         self.dropout_keep: 1.0}
            # print(X_item)
            scores = self.sess.run((self.positive), feed_dict=feed_dict)
            scores = scores.reshape(len(X_user))
            scorelist = np.append(scorelist, scores)
            # scorelist.append(scores)
            j = j + self.batch_size
        return scorelist


if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('ONCF.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    args = parse_args()
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print(
            "FM: dataset=%s, "
            "factors=%d,  "
            "#epoch=%d, "
            "batch=%d, "
            "lr=%.4f, "
            "lambda=%.1e,"
            "optimizer=%s, "
            "batch_norm=%d, "
            "keep=%.2f"
            % (args.dataset,
               args.hidden_factor,
               args.epoch,
               args.batch_size,
               args.lr,
               args.lamda,
               args.optimizer,
               args.batch_norm,
               args.keep_prob))

    save_file = 'pretrain-FM-%s/%s_%d' % (args.dataset, args.dataset, args.hidden_factor)
    # Training
    t1 = time()
    model = ONCF(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, args.epoch,
                 args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm, args.verbose)
    # model.evaluate()
    print("begin train")
    model.train(data.Train_data)
    print("end train")
    model.evaluate()
    print("finish")