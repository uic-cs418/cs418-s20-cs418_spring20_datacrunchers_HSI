# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 22:37:00 2020

@author: Varun
"""

import os
from os import listdir
from os.path import isfile, join
import struct
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import gzip
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
codepath=r'C:\Users\Varun\Desktop\IDS Project\Codes'
datapath=r'C:\Users\Varun\Desktop\IDS Project\Dataset'
os.chdir(datapath)
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
onlyfiles
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Deaths_df=pd.read_csv(r"MEASURESOFBIRTHANDDEATH.csv")
riskdf=pd.read_csv(r"RISKFACTORSANDACCESSTOCARE.csv")
Demodf=pd.read_csv(r"DEMOGRAPHICS.csv")
Demodf=Demodf[Demodf['CHSI_State_Name']=='Illinois']
riskdf=riskdf.replace(-1111.1,np.NaN)
Illinoisdf=riskdf[riskdf['CHSI_State_Name']=='Illinois']
phys_level=pd.DataFrame(Illinoisdf[['CHSI_County_Name','Prim_Care_Phys_Rate']])
Deaths_IL=Deaths_df[Deaths_df['CHSI_State_Name']=='Illinois']
deaths_perpop=Deaths_IL.merge(Demodf['Population_Size'], how='left', on=phys_level['CHSI_County_Name'])
death_ratio=np.array([x/y*100 for x,y in zip(deaths_perpop['Total_Deaths'],deaths_perpop['Population_Size'])])
phys_level['Death_Ratio']=death_ratio
X=X=phys_level[['Prim_Care_Phys_Rate','Death_Ratio']].to_numpy()
model = KMeans(n_clusters=3, random_state=1)
model.fit(X)
pred = model.predict(X)
plt.rcParams["figure.figsize"] = (15,5)

list2=[]
list1=list(model.labels_)
for i in list1:
    if i==0:
        list2.append("Cluster1")
    if i==1:
        list2.append("Cluster2")
    if i==2:
        list2.append("Cluster3")
        
ax1=sns.scatterplot(X[:,0], X[:,1], hue=list2, s=50)
ax2=plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='gold', marker='*', s=250, label='Cluster center')
ax1.set_title("Accessibility to Healthcare to Death Ratio Plot : Illinois", fontsize=15)
plt.xlabel('Physicians per 100k population')
plt.ylabel('Death Ratio')
plt.legend()
ax1.text(180,12, "Ratio of Deaths Vs Physician Rate\n have been clustered by Kmeans, an inversely\n proportional relationship shows that staffing of  medical\n depts is one of the top concerns.\n Cluster 3 has the maximum death ratio,\n and indeed the rate of primary \ncare physicians plays an important role. ", ha="left", va="top")
