#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from keras.utils import np_utils #数据预处理为0~1
import numpy as np
from sklearn.preprocessing import OneHotEncoder # 利用sklearn
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os


#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
enc = OneHotEncoder()
char_set=np.array(['3','4','6','7','8','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','T','U','V','X','W','a','b','c','d','e','f','h','i','j','k','m','n','p','r','t','u','v','w','x','y'],dtype=np.str)
print(char_set.shape)
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(char_set)
print(encoded_Y)
onehot_char=np_utils.to_categorical(encoded_Y,46)
print(onehot_char)

char_dict={}
for index, item in enumerate(onehot_char):
    char_dict[char_set[index]]=item


n=0
list_dirs = os.walk("data")
for root, dirs, files in list_dirs:
    for f in files: 
        n=n+1

data_imgs=np.zeros((n,96,96,3), dtype=np.uint8)
data_ans=[np.zeros((n,46), dtype=np.uint8),np.zeros((n,46), dtype=np.uint8),np.zeros((n,46), dtype=np.uint8),np.zeros((n,46), dtype=np.uint8)]

index=0
list_dirs = os.walk("data")
for root, dirs, files in list_dirs:
    for f in files: 
        img=os.path.join(root, f)
        img=Image.open(img)
        com_img=img.resize((96, 96))
        a=np.asarray(com_img)
        data_imgs[index,:,:]=a

        data_ans[0][index,:]=char_dict[root[-4]]
        data_ans[1][index,:]=char_dict[root[-3]]
        data_ans[2][index,:]=char_dict[root[-2]]
        data_ans[3][index,:]=char_dict[root[-1]]

        index=index+1


#np.random.permutation 函数，我们可以获得打乱后的行号
permutation = np.random.permutation(data_imgs.shape[0])
shuffled_dataset = data_imgs[permutation, :, :]


num=int(shuffled_dataset.shape[0]*0.9)
x_train = shuffled_dataset[:num]
y_train=[data_ans[i][permutation][:num] for i in range(4)]

x_test = shuffled_dataset[num:]
y_test=[data_ans[i][permutation][num:] for i in range(4)]

np.savez('mydataset',x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
