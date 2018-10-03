# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Header files
import pandas as pd
import matplotlib.pyplot as plt

data_ad=pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N=10000
d=10

ad_selected_thompson=[]
number_of_rewards_1=[0]*d
number_of_rewards_0=[0]*d
total_rewards=0

for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1,number_of_rewards_0[i]+1)              
       
        #To update the max random
        if random_beta>max_random:
            max_random=random_beta
            ad=i
            
    ad_selected_thompson.append(ad)
    reward=data_ad.values[n,ad] #compare the output we got 
    if reward==1:
        number_of_rewards_1[ad]=number_of_rewards_1[ad]+1
    else:
        number_of_rewards_0[ad]=number_of_rewards_0[ad]+1
        
    total_rewards=total_rewards+reward
        
#random.betavariate(alpha, beta)
#Beta distribution. Conditions on the parameters are alpha > 0 and beta > 0.
# Returned values range between 0 and 1.
#   https://stats.stackexchange.com/questions/47771/what-is-the-intuition-behind-beta-distribution
    
    
    
#Visualising the result:
        
plt.hist(ad_selected_thompson)        
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.title('Histogram of ads selections')
plt.show()
                                         

