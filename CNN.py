from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

dataset_path='data/all/'
n_class=2
epochs=5
batchsize=256
#todo 读取所有样本

x=[]#建立样本列表
y=[]#建立标注列表
for file in os.listdir(dataset_path):
    img=Image.open(dataset_path+file)#读取照片
    img=np.array(img)
    label=int(str(file).split('_')[0])#基于文件名提取标签
    x.append(img)
    y.append(label)


X=np.array(x)#归一化
y=np.array(map(int,y))

x_train,x_test,y_train,y_test=train_test_split((X, y))
y_train=np.array(y)


#归一化和数据reshape
x_train = x_train.astype('float32')/255.0#样本集多少不知道时就用-1 代表全部


#转onehot
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train, num_classes=10)


model=Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    input_shape=(224,224,3),
    activation='relu'
))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation='relu'
))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters=128,
    kernel_size=(3,3),
    activation='relu'
))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(n_class,activation='softmax'))

model.compile(optimizer=Adam(),loss=categorical_crossentropy,metrics=['accuracy'])

history=model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batchsize,
    shuffle=True,
    validation_split=0.1
    )
model.save('fruitOCR.h5')



#loss输出

plt.plot(history.history['loss'],label='train_loss')#plol划线 前一个括号里的是固定的，label可以自己改
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['accuracy'],label='train_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend(['loss','val_loss','acc','val_acc'],loc='upper left')#图注。按顺序来。loc是位置（左上角）

plt.savefig('accloss.jpg')