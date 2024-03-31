from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
import numpy as np
import os
from PIL import Image

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
model.summary()

def vgg16():
    input=Input(shape=(224,224,3))
    x=input
    blocks=[2,2,3,3,3]
    for block, convs in enumerate(blocks):
        #print(block, convs)
        if block == 4:
            for conv in range(convs):
                x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        else:
            for conv in range(convs):
                x = Conv2D(64*2**block, (3, 3), padding='same', activation='relu')(x)
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    outputs = Dense(1000, activation='softmax')(x)
    model = Model(inputs=input, outputs=outputs)
    return model

#vgg16().summary()