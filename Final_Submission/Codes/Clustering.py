# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 20:45:29 2020

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
from sklearn.preprocessing import normalize

datapath=r'C:\Users\Varun\Desktop\IDS Project\Dataset'
os.chdir(datapath)
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
df=pd.read_csv(r"PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.csv")
#'CHSI_County_Name_x','CHSI_State_Name_x'

LeadingCauseCols=['ALE','Diabetes','Lung_Cancer','Lung_Cancer%','Brst_Cancer%','Brst_Cancer','Col_Cancer','Col_Cancer%','MVA','Stroke', 'Suicide','CHD']
data=df[LeadingCauseCols]
data = data.fillna(0)
#df.fillna(df.mean())
data = data.astype(float)
#data=pd.to_numeric(pd.DataFrame(data))
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
plt.axhline(y=3.5, color='r', linestyle='--')

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
prd=cluster.fit_predict(data_scaled)

prd=list(prd)
result=df[['CHSI_County_Name_x','CHSI_State_Name_x']+LeadingCauseCols]
result['Cluster']=prd

plt.figure(figsize=(10, 7))  
plt.scatter(data_scaled['ALE'], data_scaled['Stroke'], c=cluster.labels_) 

result.to_csv(r'clustering_1.csv', index = False)
