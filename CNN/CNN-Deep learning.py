# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 09:24:59 2018

@author: avaithil
"""


from keras.models import Sequential
from keras.layers import Convolution2D#2D to deal with images
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

    #initialising the CNN

classifier=Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))


    #32--> Feature selection
    #3*3 matrix-->row and column


#Step 2 : Max Pooling
#To reduce the size of feature map

classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step 3: Flattening
    #to convert it inot input vector for prediction using neural network
classifier.add(Flatten())

#Step 4: Full Connetion
    #similar to classic ANN with input layer, hidden layer, output layer.
    #units-->output_dim

classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))

#Compiling the CNN

    