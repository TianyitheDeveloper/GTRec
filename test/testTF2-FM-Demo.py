import numpy as np
import tensorflow as tf


user_features = [[1,3,4,7,8],
                 [0,3,5,7,9],
                 [0,2,4,6,9]]
item_features = [[2,4,5,7,9],
                 [0,3,5,6,11],
                 [1,2,5,7,10]]

all_weights = dict()
all_weights['user_feature_embeddings'] = tf.Variable(np.random.random([10, 4]),name='user_feature_embeddings')
all_weights['item_feature_embeddings'] = tf.Variable(np.random.random([12, 4]),name='item_feature_embeddings')
all_weights['user_feature_bias'] = tf.Variable(np.random.random([10, 1]), name='user_feature_bias')
all_weights['item_feature_bias'] = tf.Variable(np.random.random([12, 1]), name='item_feature_bias')

user_feature_embeddings = tf.nn.embedding_lookup(all_weights['user_feature_embeddings'], user_features)
positive_feature_embeddings = tf.nn.embedding_lookup(all_weights['item_feature_embeddings'], item_features)

summed_user_emb = tf.reduce_sum(user_feature_embeddings, 1)
summed_item_positive_emb = tf.reduce_sum(positive_feature_embeddings, 1)
summed_positive_emb=tf.add(summed_user_emb,summed_item_positive_emb)
summed_positive_emb_square = tf.square(summed_positive_emb)

squared_user_emb=tf.square(user_feature_embeddings)
squared_item_positiv_emb=tf.square(positive_feature_embeddings)
squared_user_emb_sum=tf.reduce_sum(squared_user_emb, 1)
squared_item_positive_emb_sum = tf.reduce_sum(squared_item_positiv_emb, 1)
squared_positive_emb_sum=tf.add(squared_user_emb_sum,squared_item_positive_emb_sum)

print(summed_positive_emb_square)
