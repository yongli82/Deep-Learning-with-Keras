from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# 固定随机数种子，使每次产生的随机数保持一致
np.random.seed(1671)  # for reproducibility

# network and training
# 世代
NB_EPOCH = 200
# 批量
BATCH_SIZE = 128
# 显示详细日志  adj. 冗长的；啰嗦的
VERBOSE = 1
# 输出分类数量 这里等于 10个数字
NB_CLASSES = 10   # number of outputs = number of digits
# 优化器 SGD
OPTIMIZER = SGD() # SGD optimizer, explained later in this chapter
#
N_HIDDEN = 128
# 训练数据分割比例
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION

# data: shuffled and split between train and test sets
# 加载已经预置的mnist训练和测试数据集
# 其中X是数字图形的灰度二维矩阵集合，Y是对应的数字 0~9
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
# 本来每个图像是 28 x 28的二维矩阵，这里把它转换给一维序列
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
# 转换成浮点表示
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize 
# 把特征数据从0~255的灰度值归一到 0 ~ 1.0的浮点数
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 把输出值从数字转换成分类数据，避免0~9的数值大小的相关影响
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10 outputs
# final stage is softmax

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
