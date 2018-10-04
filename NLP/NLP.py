# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:09:10 2018

@author: avaithil

Natural Language Processing
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
#tsv means tab seperated value
#we Should use tsv for NLP. Since in text people will use comma.
data_review=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#Cleaning the tests like adjective, preprosition and article, punchuation,
#Stemming will make the word into root word.
#loves,love,loved,loving -->stemming


#Step 1: 
import re

#To remove everything except a-z , A-Z and space.
review=re.sub('[^a-zA-Z]',' ',data_review['Review'][0])

