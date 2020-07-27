"""
使用BPR损失的FM模型

2020年7月15日 17点36分
"""
import os
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.base import BaseEstimator,TransformerMixin
from time import time
import argparse
import LoadData as DATA
from tensorflow.keras.layers import BatchNormalization as batch_norm
import logging
tf.compat.v1.disable_v2_behavior()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    """
    设置命令行参数
    :return:
    """
    parser = argparse.ArgumentParser(description="Run FM")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='输入数据路径（默认为Data文件夹）')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='前k个物品组成的列表（默认为10）')
    parser.add_argument('--dataset', nargs='?', default='lastfm',
                        help='选择一个数据集（默认为lastfm，可选frappe）')
    parser.add_argument('--epoch', type=int, default=500,
                        help='设置轮次（默认为500次）')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='预训练：1为使用预训练，0为随机初始化，-1为将本模型作为预训练模型保存（默认为-1）')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='设置批量大小（默认为256）')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='设置隐向量维度（默认为64）')
    parser.add_argument('--lamda', type=float, default=0,
                        help='二阶线性部分的正则化参数（默认为0）')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout保留率（默认为0.8）')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='设置学习率（默认为0.01）')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='设置损失类型（默认为log_loss，可选square_loss）')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='设置优化器（默认为AdagradOptimizer，可选AdamOptimizer，GradientDescentOptimizer）')
    parser.add_argument('--verbose', type=int, default=1,
                        help='设置查看结果的轮次间隔（默认为1）')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='是否执行批归一化（默认为0，可选1）')
    return parser.parse_args()

class FM(BaseEstimator,TransformerMixin):
    def __init__(self,user_field_M,
                 item_field_M,
                 pretrain_flag,
                 save_file,
                 hidden_factor,
                 loss_type,
                 epoch,
                 batch_size,
                 learning_rate,
                 lamda_bilinear,
                 keep,
                 optimizer_type,
                 batch_norm,
                 verbose,
                 random_seed=2020):
        super(FM, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self._init_graph()

    def _init_graph(self):
        """

        :return:
        """
        self.graph = tf.compat.v1.Graph()           # 定义一张运算图
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # 1.输入数据
            self.user_features = tf.placeholder(tf.int32,shape=[None,None])             # 用户特征
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])       # 正样本特征
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])       # 负样本特征
            self.dropout_keep = tf.placeholder(tf.float32)                              # dropout率
            self.train_phase = tf.placeholder(tf.bool)

            # 2.变量
            self.weights = self._initialize_weights()                                   # 初始化权重

            # 3.模型
            # 3.1 正样本
            self.user_feature_embeddings = tf.nn.embedding_lookup(
                self.weights['user_feature_embeddings'],self.user_features)      # 嵌入查找用户特征嵌入表示(1)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(
                self.weights['item_feature_embeddings'],self.positive_features)  # 嵌入查找正样本物品特征嵌入表示(2)

            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings,1)                # 用户嵌入的和1+(1)=(3)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings,1)   # 物品正样本嵌入的和1+(2)=(4)
            self.summed_positive_emb = tf.add(
                self.summed_user_emb,self.summed_item_positive_emb)                 # 正样本嵌入(3)+(4)=(5)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)   # 正样本嵌入的平方 (5)²=(6)

            self.squared_user_emb = tf.square(self.user_feature_embeddings)                     # 用户嵌入的平方 (1)²=(7)
            self.squared_item_positive_emb = tf.square(self.positive_feature_embeddings)        # 物品正样本嵌入的平方(2)²=(8)
            self.squared_user_emb_sum = tf.reduce_sum(self.squared_user_emb,1)                  # 用户嵌入的平方的和1+(7)=(9)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positive_emb,1)# 物品正样本嵌入的平方的和1+(8)=(10)
            self.squared_positive_emb_sum = tf.add(
                self.squared_user_emb_sum,self.squared_item_positive_emb_sum)   # 用户嵌入的平方的和 加 物品正样本嵌入的平方的和(9)+(10)=(11)

            self.FM_positive = 0.5*tf.subtract(self.summed_positive_emb_square,
                                               self.squared_positive_emb_sum)           # FM二阶特征组合部分(6)-(11)=(12)
            self.FM_positive = tf.nn.dropout(self.FM_positive,self.dropout_keep)        # 在FM层进行dropout

            self.Bilinear_positive = tf.reduce_sum(self.FM_positive,1,keepdims=True)    # 二阶特征组合的和1+(12)=(13)
            self.user_feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['user_feature_bias'],
                                       self.user_features),1)                       # 用户特征偏重(14)
            self.item_feature_bias_positive = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'],
                                       self.positive_features), 1)                  # 正样本特征偏重(15)
            self.positive = tf.add_n([self.Bilinear_positive,self.user_feature_bias,
                                      self.item_feature_bias_positive])             # 正样本预测值(13)+(14)+(15)

            # 3.2 负样本
            self.negative_feature_embeddings = tf.nn.embedding_lookup(
                self.weights['item_feature_embeddings'],self.negative_features)                 # 嵌入查找正样本物品特征嵌入表示(16)
            self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings,1)   # 物品负样本嵌入的和1+(16)=(17)
            self.summed_negative_emb = tf.add(
                self.summed_user_emb,self.summed_item_negative_emb)                             # 负样本嵌入(3)+(17)=(18)
            self.summed_negative_emb_square = tf.square(self.summed_negative_emb)               # 负样本嵌入的平方 (18)²=(19)

            self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)        # 物品负样本嵌入的平方(17)²=(20)
            self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb,1)# 物品负样本嵌入的平方的和1+(17)=(21)
            self.squared_negative_emb_sum = tf.add(
                self.squared_user_emb_sum,self.squared_item_negative_emb_sum)   # 用户嵌入的平方的和 加 物品负样本嵌入的平方的和(9)+(21)=(22)

            self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square,
                                                 self.squared_negative_emb_sum)         # FM二阶特征组合部分(6)-(11)=(12)
            self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep)       # 在FM层进行dropout

            self.Bilinear_negative = tf.reduce_sum(self.FM_negative, 1, keepdims=True)  # 二阶特征组合的和1+(12)=(23)
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'],
                                       self.negative_features), 1)                      # 负样本特征偏重(24)
            self.negative = tf.add_n([self.Bilinear_negative, self.user_feature_bias,
                                      self.item_feature_bias_negative])                 # 负样本预测值(23)+(14)+(24)

            # 4.损失：BPR损失
            self.loss = -tf.log(tf.sigmoid(self.positive-self.negative))
            self.loss = tf.reduce_sum(self.loss)

            # 5.优化器
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # 6.初始化
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()        # 全局变量初始化
            self.sess = tf.Session()
            self.sess.run(init)                             # 执行初始化

            # 参数数量
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("参数总数为: %d" % total_parameters)


    def _initialize_weights(self):
        """
        初始化权重
        :return all_weights: 返回所有权重，包括用户特征嵌入表示，物品特征嵌入表示，用户特征偏重，物品特征偏重
        """
        all_weights = dict()            # 将所有权重用字典存储
        if self.pretrain_flag > 0:
            # 加载预训练模型权重作为初始化权重
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')     # 加载预训练模型
            pretrain_graph = tf.get_default_graph()                                 # 预训练图
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')    # 用户特征嵌入表示
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')    # 物品特征嵌入表示
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')                # 用户特征偏重
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')                # 物品特征偏重
            with tf.Session() as sess:
                weight_saver.restore(sess,self.save_file)
                ue,ie,ub,ib = sess.run(
                    [user_feature_embeddings,item_feature_embeddings,user_feature_bias,item_feature_bias]   # 执行加载操作
                )
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)          # 加载用户特征嵌入表示为变量
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)          # 加载物品特征嵌入表示为变量
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)                # 加载用户特征偏重为变量
            all_weights['item_feature_bias'] = tf.Variable(ie, dtype=tf.float32)                # 加载物品特征偏重为变量
        else:
            # 不加载预训练模型权重作为初始化权重，直接随机生成
            all_weights['user_feature_embeddings'] = tf.Variable(tf.random_normal(
                [self.user_field_M,self.hidden_factor],0.0,0.1), name='user_feature_embeddings')    # 用户特征嵌入表示
            all_weights['item_feature_embeddings'] = tf.Variable(tf.random_normal(
                [self.item_field_M,self.hidden_factor],0.0,0.1), name='item_feature_embeddings')    # 物品特征嵌入表示
            all_weights['user_feature_bias'] = tf.Variable(tf.random_normal(
                [self.user_field_M,1],0.0,0.1), name='user_feature_bias')                           # 用户特征偏重
            all_weights['item_feature_bias'] = tf.Variable(tf.random_normal(
                [self.item_field_M,1],0.0,0.1), name='item_feature_bias')                           # 物品特征偏重
        return all_weights

    def partial_fit(self,data):
        """
        拟合一个批量
        :param data: 一个批量的数据
        :return loss: 每个批量的损失
        """
        feed_dict = {
            self.user_features:data['X_user'],
            self.positive_features: data['X_positive'],
            self.negative_features: data['X_negative'],
            self.dropout_keep:self.keep,
            self.train_phase: True
        }
        loss,opt = self.sess.run((self.loss,self.optimizer),feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self,train_data,batch_size):
        X_user,X_positive,X_negative = [],[],[]
        all_items = data.binded_items.values()
        # 获取样本
        while len(X_user)<batch_size:
            index = np.random.randint(0,len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])   # 用户特征
            user_id = data.binded_users[user_features]      # 用户ID
            pos = data.user_positive_list[user_id]          # 匹配用户ID的正样本列表
            neg = np.random.randint(len(all_items))         # 匹配用户ID的负样本列表
            while (neg in pos): neg = np.random.randint(len(all_items)) # 负样本在正样本列表中则重新采样
            negative_feature = data.item_map[neg].strip().split('-')    # 负样本物品特征
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user':X_user,'X_positive':X_positive,'X_negative':X_negative}

    def train(self,Train_data):
        for epoch in range(self.epoch):
            total_loss = 0
            total_batch = int(len(Train_data['X_user'])/self.batch_size)
            for i in range(total_batch):
                # 生成一个批量
                batch_xs = self.get_random_block_from_data(Train_data,self.batch_size)
                loss = self.partial_fit(batch_xs)
                total_loss = total_loss + loss
            logger.info("第%d个轮次的损失是%f"%(epoch,total_loss))
            if (epoch+1) % 100 == 0:
                model.evaluate()
        print("结束训练，开始保存")
        if self.pretrain_flag < 0:
            print("保存模型参数以作为预训练模型")
            self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0,0,0,0]
        rank = [[],[],[],[]]
        topK = [5,10,15,20]
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
            scores = np.delete(scores,visited)
            # 是否命中
            sorted_scores = sorted(scores,reverse=True)

            label = []
            for i in range(len(topK)):
                label.append(sorted_scores[topK[i]-1])
                if true_item_score>=label[i]:
                    count[i] = count[i]+1
                    rank[i].append(sorted_scores.index(true_item_score)+1)
        for i in range(len(topK)):
            mrr = 0
            ndcg = 0
            hr = float(count[i]/len(data.Test_data['X_user']))
            for item in rank[i]:
                mrr += float(1.0)/item
                ndcg += float(1.0)/np.log2(item+1)
            mrr /= len(data.Test_data['X_user'])
            ndcg /= len(data.Test_data['X_user'])
            k = (i+1)*5
            logger.info("\n前%d个物品的HR,NDCG分别为%f,%f"%(k,hr,ndcg))

    def get_scores_per_user(self,user_features):
        X_user,X_item = [],[]
        all_items = data.binded_items.values()
        for itemID in range(len(all_items)):
            X_user.append(user_features)
            item_feature = [int(feature) for feature in data.item_map[itemID].strip().split('-')[0:]]
            X_item.append(item_feature)
        feed_dict = {
            self.user_features:X_user,
            self.positive_features:X_item,
            self.train_phase:False,
            self.dropout_keep:1.0
        }
        scores = self.sess.run((self.positive),feed_dict=feed_dict)
        scores = scores.reshape(len(all_items))
        return scores

if __name__ == '__main__':
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('fm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # 数据加载
    args = parse_args()
    data = DATA.LoadData(args.path,args.dataset)
    if args.verbose > 0:
        print(
            "\nFM模型："
            "\n数据集设置为 ",args.dataset,
            "\n隐向量维度设置为 ",args.hidden_factor,
            "\n轮次设置为",args.epoch,
            "\n批量大小设置为",args.batch_size,
            "\n学习率设置为",args.lr,
            "\n优化器设置为",args.optimizer,
            "\n二阶线性部分的正则化参数设置为",args.lamda,
            "\n批归一化设置为",("执行" if args.batch_norm else "不执行"),
            "\ndropout保留率设置为",args.keep_prob
        )
    save_file = 'pretrain-FM-%s/%s_%d'%(args.dataset,args.dataset,args.hidden_factor)

    # 训练
    t1 = time()
    model = FM(
        data.user_field_M,
        data.item_field_M,
        args.pretrain,
        save_file,
        args.hidden_factor,
        args.loss_type,
        args.epoch,
        args.batch_size,
        args.lr,
        args.lamda,
        args.keep_prob,
        args.optimizer,
        args.batch_norm,
        args.verbose
    )
    print("\n开始训练")
    model.train(data.Train_data)
    print("\nFM模型在CFM论文中效果为")
    print("Frappe数据集："
          "\nHR@5:0.4204  \nNG@5:0.3054"
          "\nHR@10:0.5486 \nNG@10:0.3469"
          "\nHR@20:0.6590 \nNG@20:0.3750"
          "\nLastfm数据集："
          "\nHR@5:0.1658  \nNG@5:0.1142"
          "\nHR@10:0.2382 \nNG@10:0.1374"
          "\nHR@20:0.3537 \nNG@20:0.1665")
    print("\n训练结束")















