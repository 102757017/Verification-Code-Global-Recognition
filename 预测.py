# coding: utf-8

#!/usr/bin/python

import numpy as np

#随机数种子不变的情况下，random.random()生成的随机数是不变的
np.random.seed(123)

from keras.models import Sequential

from keras.layers import Dense #导入全连接神经层

from keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元

from keras.layers import Activation #导入激活函数

from keras.layers import Conv2D #导入卷积层

from keras.layers import MaxPooling2D #导入池化层
from keras.layers import Reshape,UpSampling2D,Conv2DTranspose

from keras.layers import Flatten

from keras.utils import np_utils #数据预处理为0~1

from keras.datasets import mnist #导入手写数据集

from keras.models import load_model 

from matplotlib import pyplot as plt

from keras.callbacks import ReduceLROnPlateau #动态调整学习率

from keras.callbacks import ModelCheckpoint #训练途中自动保存模型
import os
import matplotlib
from PIL import Image


from keras.utils import np_utils #数据预处理为0~1
import numpy as np
from sklearn.preprocessing import OneHotEncoder # 利用sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os
enc = OneHotEncoder()
char_set=np.array(['3','4','6','7','8','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','T','U','V','X','W','a','b','c','d','e','f','h','i','j','k','m','n','p','r','t','u','v','w','x','y'],dtype=np.str)
print(char_set.shape)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(char_set)
print(encoded_Y)

onehot_char=np_utils.to_categorical(encoded_Y,46)
print(onehot_char)

r_dict={}
for index, item in enumerate(onehot_char):
    r_dict[np.argmax(onehot_char[index])]=char_set[index]


    

f=np.load('testdata.npz')
print(type(f),f)
x_test = f['x_test']

print('x_test',x_test.shape)


#因为卷积层要求输入的input_shape=(30,68,3)，因此需要将输入数据增加一个维度，变成n个(30,70,3)的数组
x_test=x_test.reshape(x_test.shape[0],96,96,3)



#将数据转换为浮点数
x_test=x_test.astype("float32")


#将输入数组中的数据转换为0~1之间的数
x_test /=255


#加载模型
model = load_model('model0.h5')

#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_test[0:5],batch_size=32,verbose=1)

l0=[r_dict[np.argmax(p)] for p in y_predict[0]]
l1=[r_dict[np.argmax(p)] for p in y_predict[1]]
l2=[r_dict[np.argmax(p)] for p in y_predict[2]]
l3=[r_dict[np.argmax(p)] for p in y_predict[3]]

y_predict=[str(l0[i])+str(l1[i])+str(l2[i])+str(l3[i]) for i in range(len(l0))]

img1=Image.fromarray(np.uint8(x_test[0]*255))
img1.show()
print(y_predict[0])



