"""
数据集的读取，数据预处理，负采样

2020年7月15日 17点36分
"""

import numpy as np
import os

class LoadData(object):

    def __init__(self,path,dataset):
        self.path = path + dataset + "/"            # 数据路径
        self.trainfile = self.path + "train.csv"    # 读取训练集
        self.testfile = self.path + "test.csv"      # 读取测试集
        self.user_field_M, self.item_field_M = self.get_length()    # 分别获取用户和物品的特征域数
        print("用户特征数：", self.user_field_M)
        print("物品特征数：", self.item_field_M)
        print("所有特征数",self.user_field_M+self.item_field_M)
        self.item_bind_M = self.bind_item()         # 绑定物品编号
        self.user_bind_M = self.bind_user()         # 绑定用户编号
        print("为特定的物品上下文分配物品ID，数量为", len(self.binded_items.values()))
        print("为特定的用户上下文分配用户ID，数量为", len(self.binded_users.values()))
        self.user_positive_list = self.get_positive_list(self.trainfile)        # 采样
        self.Train_data , self.Test_data = self.construct_data()                # 构建可用数据

    def get_length(self):
        """
        获取用户和物品数量最大值
        :return: 用户数量，物品数量
        """
        length_user = 0
        length_item = 0
        f = open(self.trainfile)
        line = f.readline()
        while line:
            user_features = line.strip().split(',')[0].split('-')
            item_features = line.strip().split(',')[1].split('-')
            for user_feature in user_features:
                feature = int(user_feature)
                if feature > length_user:
                    length_user = feature
            for item_feature in item_features:
                feature = int(item_feature)
                if feature > length_item:
                    length_item = feature
            line = f.readline()
        f.close()
        return length_user+1, length_item+1

    def bind_item(self):
        self.binded_items = {}
        self.item_map = {}
        self.bind_i(self.trainfile)
        self.bind_i(self.testfile)
        return len(self.binded_items)

    def bind_i(self,file):
        f = open(file)
        line = f.readline()
        i = len(self.binded_items)
        while line:
            features = line.strip().split(',')
            item_features = features[1]
            if item_features not in self.binded_items:
                self.binded_items[item_features] = i
                self.item_map[i] = item_features
                i = i + 1
            line = f.readline()
        f.close()


    def bind_user(self):
        self.binded_users = {}
        self.bind_u(self.trainfile)
        self.bind_u(self.testfile)
        return len(self.binded_users)

    def bind_u(self,file):
        f = open(file)
        line = f.readline()
        i = len(self.binded_users)
        while line:
            features = line.strip().split(',')
            user_features = features[0]
            if user_features not in self.binded_users:
                self.binded_users[user_features] = i
                i = i + 1
            line = f.readline()
        f.close()

    def get_positive_list(self,file):
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            features = line.strip().split(',')
            user_id = self.binded_users[features[0]]
            item_id = self.binded_items[features[1]]
            if user_id in user_positive_list:
                user_positive_list[user_id].append(item_id)
            else:
                user_positive_list[user_id] = [item_id]
            line = f.readline()
        f.close()
        return user_positive_list

    def construct_data(self):
        """
        构建可用训练数据和测试数据
        :return: 训练数据，测试数据
        """
        X_user, X_item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_user,X_item)
        print("训练集数量：",len(X_user))

        X_user, X_item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_user, X_item)
        print("测试集数量：", len(X_user))
        return Train_data,Test_data

    def read_data(self,file):
        f = open(file)
        X_user = []
        X_item = []
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_features = features[0].split('-')
            X_user.append([int(item) for item in user_features[0:]])
            item_features = features[1].split('-')
            X_item.append([int(item) for item in item_features[0:]])
            line = f.readline()
        f.close()
        return X_user,X_item

    def construct_dataset(self, X_user, X_item):
        Data_Dic = {}
        indexs = range(len(X_user))
        Data_Dic['X_user'] = [X_user[i] for i in indexs]
        Data_Dic['X_item'] = [X_item[i] for i in indexs]
        return Data_Dic



# print("-"*20)
# DATAlastfm = LoadData('Data/','lastfm')