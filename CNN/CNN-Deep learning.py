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

