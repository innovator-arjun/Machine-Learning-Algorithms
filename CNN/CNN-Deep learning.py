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

#Part-1 Building the CNN

    #initialising the CNN

classifier=Sequential()

#Step 1: Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

    #32--> Feature selection
    #3*3 matrix-->row and column


#Step 2 : Max Pooling
#To reduce the size of feature map

classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolutional layer to improve accuracy
classifier.add(Convolution2D(32,3,3,activation='relu'))
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

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#more than two outcomes loss='categorical_crossentropy'

#Part-2 Fitting the CNN to the images- 
#image agumentation has to done to prevent over fitting. when we have less data
#check the link for the below code--> https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
                            training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)
