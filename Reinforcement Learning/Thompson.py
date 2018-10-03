# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_ad=pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N=10000
d=10
ads_selected=[]
total_reward=0

for n in range(0,N):
    ad=random.randrange(d)
    ads_selected.append(ad)
    reward=data_ad.values[n,ad]
    total_reward=total_reward+reward
    
    
plt.hist(ads_selected)
plt.title('Histogram of random selection')
plt.xlabel('Ad type')
plt.ylabel('Count')

ad_selected_thompson=[]
number_of_rewards_1=[0]*d
number_of_rewards_0=[0]*d
total_rewards=0

#Theta i is the porb of success and it is prior distribution


