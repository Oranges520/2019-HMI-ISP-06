import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os, glob

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

train_img_list = glob.glob("fingers/train/*.png")
test_img_list = glob.glob("fingers/test/*.png")
print(len(train_img_list),
     len(test_img_list), sep = '\n')
     
def import_data():
    train_img_data = []
    test_img_data = []
    train_label_data = []
    test_label_data = []
    
    for img in train_img_list:
        img_read = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_read = cv2.resize(img_read, (128,128))
        train_img_data.append(img_read)
        train_label_data.append(int(img[-6]))
    
    for img in test_img_list:
        img_read = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img_read = cv2.resize(img_read, (128,128))
        test_img_data.append(img_read)
        test_label_data.append(int(img[-6]))
        
    return np.array(train_img_data), np.array(test_img_data), np.array(train_label_data), np.array(test_label_data)
    
xtrain, xtest, ytrain, ytest = import_data()

ytrain.shape

from keras.utils import to_categorical 
xtrain = xtrain.reshape(xtrain.shape[0], 128, 128, 1)
xtest = xtest.reshape(xtest.shape[0], 128, 128, 1)

ytrain = to_categorical(ytrain, 6)
ytest = to_categorical(ytest, 6)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.40))
model.add(Dense(6, activation = 'softmax'))

model.summary()
