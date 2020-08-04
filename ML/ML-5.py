"""
机器学习实例第五章推荐系统

2020年8月4日 09点33分
"""
import numpy as np
from functools import reduce
from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.pipeline import Pipeline
import json


# -------------------------为数据处理构建函数组合-------------------------
def add3(input):
    return map(lambda x:x+3,input)  # python3中的map函数返回迭代器，注意用list转换

def mul2(input):
    return map(lambda x:x*2,input)

def sub5(input_array):
    return map(lambda x: x-5, input_array)

def funtion_composer(*args):
    """
    函数组合器
    :param args: 输入函数
    :return: 合并后函数
    """
    return reduce(lambda f,g:lambda x:f(g(x)),args)

# -------------------------构建机器学习流水线-------------------------
X,y = samples_generator.make_classification(
    n_informative=4,n_features=20,n_redundant=0,random_state=5)     # 生成示例数据(100x20)
selector_k_best = SelectKBest(f_regression,k=10)                    # 特征选择器
classifier = RandomForestClassifier(n_estimators=50,max_depth=4)    # 随机森林分类器
pipeline_classifier = Pipeline([('selector',selector_k_best),('rf',classifier)])    # 机器学习流水线
pipeline_classifier.set_params(selector__k=6,rf__n_estimators=25)   # 更新参数
pipeline_classifier.fit(X,y)                                        # 拟合
prediction = pipeline_classifier.predict(X)
features_status = pipeline_classifier.named_steps['selector'].get_support()
selector_features = []
for count, item in enumerate(features_status):
    if item:
        selector_features.append(count)

# -------------------------计算欧氏距离分数-------------------------
def euclidean_score(dataset, user1, user2):
    """
    计算欧式距离分数
    :param dataset: 数据集
    :param user1: 用户1
    :param user2: 用户2
    :return: 欧氏距离分数
    """
    if user1 not in dataset:
        raise TypeError('用户' + user1 + '不在数据集中')
    if user2 not in dataset:
        raise TypeError('用户' + user2 + '不在数据集中')
    rated_by_both = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1         # 被user1和user2共同评分过的电影
    if len(rated_by_both) == 0:
        return 0                            # 如果两个用户没有看过相同电影，则欧式距离分数自然为0
    squared_differences = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))  # 对每个共同评分计算平方和的平方根并归一化

# -------------------------计算皮尔逊相关系数-------------------------
def pearson_score(dataset, user1, user2):
    """
    计算皮尔逊相关系数
    :param dataset: 数据集
    :param user1: 用户1
    :param user2: 用户2
    :return: 皮尔逊相关系数
    """
    if user1 not in dataset:
        raise TypeError('用户' + user1 + '不在数据集中')
    if user2 not in dataset:
        raise TypeError('用户' + user2 + '不在数据集中')
    rated_by_both = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1         # 被user1和user2共同评分过的电影
    if len(rated_by_both) == 0:
        return 0                            # 如果两个用户没有看过相同电影，则皮尔逊相关系数为0

    # user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])    # r_u
    # user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])    # r_v
    # user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    # user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])
    # product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])
    # Sxy = product_sum - (user1_sum * user2_sum / len(rated_by_both))
    # Sxx = user1_squared_sum - np.square(user1_sum) / len(rated_by_both)
    # Syy = user2_squared_sum - np.square(user2_sum) / len(rated_by_both)
    # if Sxx * Syy == 0:
    #     return 0
    # return Sxy / np.sqrt(Sxx * Syy)

    ru_hat = np.sum([dataset[user1][item] for item in rated_by_both]) / len(rated_by_both)
    rv_hat = np.sum([dataset[user2][item] for item in rated_by_both]) / len(rated_by_both)

    fenzi = np.sum([(dataset[user1][item]-ru_hat)*(dataset[user2][item]-rv_hat) for item in rated_by_both])

    fenmu1 = np.sum([(np.square(dataset[user1][item]- ru_hat)) for item in rated_by_both])
    fenmu2 = np.sum([(np.square(dataset[user2][item]- rv_hat)) for item in rated_by_both])

    return fenzi/np.sqrt(fenmu1*fenmu2)



if __name__ == '__main__':
    print("-------------------------为数据处理构建函数组合-------------------------")
    arr = np.array([2,3,4,5,6,7])
    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print("不使用函数组合",list(arr3))
    funtion_composed = funtion_composer(sub5,mul2,add3)
    print("使用函数组合",list(funtion_composed(arr)))

    print("-------------------------构建机器学习流水线-------------------------")
    print("预测:", prediction)
    print("真实:", y)
    print("对比:", prediction ^ y)
    print("分数", pipeline_classifier.score(X, y))
    print("选择的特征", selector_features)

    print("-------------------------计算欧氏距离分数-------------------------")
    with open('movie_ratings.json','r') as f:
        data = json.loads(f.read())

    user1 = 'John Carson'
    user2 = 'Michelle Peterson'
    print("%s和%s的欧氏距离分数为%f" % (user1, user2, euclidean_score(data, user1, user2)))
    print("-------------------------计算皮尔逊相关系数-------------------------")
    print("%s和%s的皮尔逊相关系数为%f" % (user1, user2, pearson_score(data, user1, user2)))

