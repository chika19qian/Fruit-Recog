from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
import numpy as np
import os
from PIL import Image
from train_vgg16 import vgg16


dataset_path='data/all'
n_class=2
epochs=5
batchsize=256

x=[]#建立样本列表
y=[]#建立标注列表
for file in os.listdir(dataset_path):

    img=Image.open(dataset_path+file)#读取照片
    img=np.array(img)
    label=int(str(file).split('_')[0])#基于文件名提取标签
    x.append(img)
    y.append(label)


x_train=np.array(x).astype('float32')/255.0#归一化
y_train=np.array(y)
y_train=to_categorical(y=y_train,num_classes=n_class)
print(y_train.shape)

model=Sequential()

inputs=Input(shape=(224, 224, 3))
#block1
x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#block2
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#block3
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#block4
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#block5
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = Conv2D(512, (3, 3), padding='same',activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#block6
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(),loss=categorical_crossentropy,metrics=['accuracy'])

history=model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batchsize,
    shuffle=True,
    validation_split=0.1
    )
model.save('fruitVGG16.h5')



#loss输出

plt.plot(history.history['loss'],label='train_loss')#plol划线 前一个括号里的是固定的，label可以自己改
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['accuracy'],label='train_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend(['loss','val_loss','acc','val_acc'],loc='upper left')#图注。按顺序来。loc是位置（左上角）

plt.savefig('accloss.jpg')