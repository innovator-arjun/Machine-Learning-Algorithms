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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#importing the dataset
#tsv means tab seperated value
#we Should use tsv for NLP. Since in text people will use comma.
data_review=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)



#Step 1: 


#To remove everything except a-z , A-Z and space.
review=re.sub('[^a-zA-Z]',' ',data_review['Review'][0])

#Step 2:

#to make the text to lower case
review=review.lower()


#Cleaning the tests like adjective, preprosition and article, punchuation,
#Stemming will make the word into root word.
#loves,love,loved,loving -->stemming

#Step 3:

nltk.download('stopwords')

#to make the string to list of strings
review=review.split()

#for loop to remove the stop words from our review
#use set() to commpute fastter
ps=PorterStemmer()
review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

#to make it as a string
review=' '.join(review)

#create a for loop to clean all the review
corpus=[]
for i in range(0,1000):
    comment=re.sub('[^a-zA-Z]',' ',data_review['Review'][i])
    comment=comment.lower()
    comment=comment.split()
    ps=PorterStemmer()
    comment=[ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment=' '.join(comment)
    corpus.append(comment)

#Create the bag of words model
#loved,loves,love-->love

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
#max_feature to select the frequent words
X=cv.fit_transform(corpus).toarray() # To create a sparse matrix based on the words

#we need to include dependent variable

y=data_review.iloc[:,1].values































