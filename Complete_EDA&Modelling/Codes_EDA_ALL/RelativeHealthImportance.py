# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:53:26 2020

@author: Varun
"""
#!/usr/bin/python
##########importing libraries###################
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
###########changing working directory################################
mypath=r'D:\Drive\Coursework\IDS\Project_IDS\Dataset'
os.chdir(mypath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
####################DEMOGRAPHICS#########################################
##descriptive statistics
df_RHI=pd.read_csv('RELATIVEHEALTHIMPORTANCE.csv')
df_RHI=df_RHI.loc[df_RHI['CHSI_State_Name']=='Illinois']
df_RHI.columns
df_RHI.rename(columns={
'RHI_LBW_Ind':'low birth wt less than 2000g',
'RHI_VLBW_Ind':'very low birth wt less than 1500g',
'RHI_Premature_Ind':'premature births less than 37weeks',
'RHI_Under_18_Ind':' births to women under 18',
'RHI_Over_40_Ind':'births to women over 40',
'RHI_Unmarried_Ind':'births to unmarried women',
'RHI_Late_Care_Ind':'no care in first trimester',
'RHI_Infant_Mortality_Ind':'infant mortality',
'RHI_IM_Wh_Non_Hisp_Ind':'White non Hispanic infant mortality',
'RHI_IM_Bl_Non_Hisp_Ind':'Black non Hispanic infant mortality',
'RHI_IM_Hisp_Ind':'Hispanic infant mortality',
'RHI_IM_Neonatal_Ind':'neonatal infant mortality',
'RHI_IM_Postneonatal_Ind':'post-neonatal infant mortality',
'RHI_Brst_Cancer_Ind':'breast cancer (female)',
'RHI_Col_Cancer_Ind':'colon cancer',
'RHI_CHD_Ind':'coronary heart disease',
'RHI_Homicide_Ind':'homicide',
'RHI_Lung_Cancer_Ind':'lung cancer',
'RHI_MVA_Ind':'motor vehicle injuries',
'RHI_Stroke_Ind':'stroke',
'RHI_Suicide_Ind':'suicide',
'RHI_Injury_Ind':'unintentional injury'}
,inplace=True)
ListofNans=[-9999,-2222,-2222.2,-2,-1111.1,-1111,-1,-9998.9]
df_RHI=df_RHI.replace([i for i in ListofNans], np.NAN)
UsefulCols= ['low birth wt less than 2000g', 'very low birth wt less than 1500g',
       'premature births less than 37weeks', ' births to women under 18',
       'births to women over 40', 'births to unmarried women',
       'no care in first trimester', 'infant mortality',
       'White non Hispanic infant mortality',
       'Black non Hispanic infant mortality', 'Hispanic infant mortality',
       'neonatal infant mortality', 'post-neonatal infant mortality',
       'breast cancer (female)', 'colon cancer', 'coronary heart disease',
       'homicide', 'lung cancer', 'motor vehicle injuries', 'stroke',
       'suicide', 'unintentional injury']
df_RHI[UsefulCols]=df_RHI[UsefulCols].replace({1:'No',
2:'Yes',
3:'Like Peers',
4:'Unlike Peers',
5:'Like Peers Like US',
6:'Like Peers Unlike US',
7:'Unlike Peers Like US',
8:'Unlike Peers Unlike US',
})
##############JOIN TABLES##################
#RELATIVEHEALTHIMPORTANCE & DEMOGRAPHICS
df_RHI.head()
df_RHI.columns
