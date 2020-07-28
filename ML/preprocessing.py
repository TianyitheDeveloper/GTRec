import numpy as np
from sklearn import preprocessing

data = np.array([[3,  -1.5,    2,  -5.4],
                 [0,    4,    -0.3,  2.1],
                 [1,    3.3,   -1.9, -4.3]])

data_1 = preprocessing.scale(data)
print(data_1)
data_2 = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(data)
print(data_2)
data_3 = preprocessing.normalize(data,norm='l1')
print(data_3)
data_4 = preprocessing.Binarizer(threshold=1.4).transform(data)
print(data_4)

label_encoder = preprocessing.LabelBinarizer()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
for i, item in enumerate(label_encoder.classes_):
    print(item,"---",i)

