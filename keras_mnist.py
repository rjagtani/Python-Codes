# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:41:35 2018

@author: rjagtani
"""

import keras
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.datasets import mnist
from timeit import default_timer as timer

(train_x,train_y),(test_x,test_y)=mnist.load_data()

'''os.getcwd()
os.chdir('C:\\Users\\rjagtani\\Desktop\\Python')
titanic=pd.read_csv('train_titanic.csv')
titanic_predictors=titanic.loc[:,['Sex','Fare','Age']]
titanic_predictors=pd.get_dummies(titanic_predictors)
titanic_predictors.Age=titanic_predictors.Age.fillna(28)'''

predictors=train_x.reshape(60000,784)
target=to_categorical(train_y)
model=Sequential()
model.add(Dense(20,activation='relu',input_shape=(784,)))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
start=timer()
model.fit(predictors,target,epochs = 10,batch_size=10,validation_split = 0.2)
end=timer()
print(end-start)
model.summary()
model.save('first_nn.h5')
model=load_model('first_nn.h5')
