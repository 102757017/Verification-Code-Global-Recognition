#!/usr/bin/python
# # -*- coding: UTF-8 -*-
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.models import Model #导入函数式模型
from keras.layers import Input #导入输入数据层
import os
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D #导入池化层
from keras.layers import AveragePooling2D #导入池化层
from keras.layers import Dense #导入全连接神经层
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元
from keras.layers import Activation #导入激活函数
import keras
from keras.layers import K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ReduceLROnPlateau #动态调整学习率
from keras.callbacks import ModelCheckpoint #训练途中自动保存模型
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.utils import plot_model
from matplotlib import pyplot as plt


#include_top=True，完整的模型
#include_top=False，去掉最后的3个全连接层，用来做fine-tuning专用，专门开源了这类模型。
#迁移学习必须指定输入图片的shape，否则会默认为(224, 224)
base_model = MobileNetV2(input_shape=(96,96,3),weights='imagenet',include_top=False)

#函数式模型
inputs1 = Input(shape=(96, 96, 3))


# 增加MobileNetV2层
x1=base_model(inputs1)
x1=Flatten()(x1)

m1=Dense(300,activation='relu')(x1)
m2=Dense(300,activation='relu')(x1)
m3=Dense(300,activation='relu')(x1)
m4=Dense(300,activation='relu')(x1)


midputs = Input(shape=(None,300))
y = Dense(46,activation='softmax')(midputs)
model2=Model(inputs=midputs, outputs=y)

y1 = model2(m1)
y2 = model2(m2)
y3 = model2(m3)
y4 = model2(m4)



# 预训练模型与新加层的组合
model = Model(inputs=inputs1, outputs=[y1,y2,y3,y4])

# 只训练新加的Top层，冻结MobileNet所有层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"G:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"
plot_model(model, to_file='模型结构.png',show_shapes=True)


f=np.load('mydataset.npz')
print(type(f),f)
x_train = f['x_train']
y_train = f['y_train']
x_test = f['x_test']
y_test = f['y_test']
print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)

#因为卷积层要求输入的input_shape=(30,68,3)，因此需要将输入数据增加一个维度，变成n个(30,70,3)的数组
x_train=x_train.reshape(x_train.shape[0],96,96,3)
x_test=x_test.reshape(x_test.shape[0],96,96,3)

#将数据转换为浮点数
x_train=x_train.astype('float32')
y_train=y_train.astype("float32")
x_test=x_test.astype("float32")
y_test=y_test.astype("float32")

#将输入数组中的数据转换为0~1之间的数
x_train /= 255
y_train /= 255
y_train=[y_train[i] for i in range(4)]
y_test=[y_test[i] for i in range(4)]


#学习率是每个batch权重往梯度方向下降的步距，学习率越高，loss下降越快，但是太高时会无法收敛到最优点（在附近打摆），keras默认的学习率是0.01
#设置动态学习率，使用回调函数调用
#monitor：被监测的量
#factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
#patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
#mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=2, mode='auto')

filepath = "weights-improvement.hdf5"
# 每个epoch确认确认monitor的值，如果训练效果提升, 则将权重保存, 每提升一次, 保存一次
#mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode='auto')

#实现断点继续训练
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

#训练模型
#batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
#nb_epochs：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为”number of”的意思
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
#回调函数为一个list,list中可有多个回调函数,回调函数以字典logs为参数,模型的.fit()中有下列参数会被记录到logs中：
##正确率和误差，acc和loss，如果指定了验证集，还会包含验证集正确率和误差val_acc和val_loss，val_acc还额外需要在.compile中启用metrics=['accuracy']。
history =model.fit(x_train,y_train,batch_size=500,epochs=5,verbose=1,callbacks=[reduce_lr,checkpoint])
#返回记录字典，包括每一次迭代的训练误差率和验证误差率


# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py

# 评估模型
#model.evaluate返回的是一个list,其中第一个元素为loss指标，其它元素为metrias中定义的指标，metrias指定了N个指标则返回N个元素
loss,accuracy = model.evaluate(x_test,y_test,batch_size=500)
print('\ntest loss',loss)
print('accuracy',accuracy)

#绘图
#acc是准确率，适合于分类问题。对于回归问题，返回的准确率为0
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['acc'], label='train_acc')
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
