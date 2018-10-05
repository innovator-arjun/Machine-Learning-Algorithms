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



#Step 1: 
import re

#To remove everything except a-z , A-Z and space.
review=re.sub('[^a-zA-Z]',' ',data_review['Review'][0])

#Step 2:

#to make the text to lower case
review=review.lower()


#Cleaning the tests like adjective, preprosition and article, punchuation,
#Stemming will make the word into root word.
#loves,love,loved,loving -->stemming

#Step 3:
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#to make the string to list of strings
review=review.split()

#for loop to remove the stop words from our review
#use set() to commpute fastter
review=[word for word in review if not word in set(stopwords.words('english'))]