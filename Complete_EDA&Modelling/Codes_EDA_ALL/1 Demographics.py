# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:34:53 2020

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
df_Demog=pd.read_csv('DEMOGRAPHICS.csv')
df_Demog.columns
#df_Demog=df_Demog.loc[df_Demog['CHSI_State_Name']=='Illinois']
df_Demog=df_Demog[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name','CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number','Population_Size','Population_Density','Poverty','Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over','White','Black', 'Native_American','Asian', 'Hispanic']]
ListofNans=[-9999,-2222,-2222.2,-2,-1111.1,-1111,-1,-9998.9]
df_Demog=df_Demog.replace([i for i in ListofNans], np.NAN)
####################################################################################

PovertyStats=df_Demog['Poverty'].describe()
print(PovertyStats)

Races_df=df_Demog[['White','Black','Asian', 'Hispanic']]
print(Races_df.describe())

Ages_df=df_Demog[['Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
print(Ages_df.describe())

ax = sns.boxplot(data=df_Demog[['Black', 'Native_American','Asian', 'Hispanic']], palette="Set2")
ax.set_yscale('log')
################################################################################

Race_Age=df_Demog[['White','Black','Asian', 'Hispanic','Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
Race_AgeCorr=pd.DataFrame(Race_Age.corr())
Race_AgeCorr=Race_AgeCorr[Race_AgeCorr.index.isin(['White','Black','Asian', 'Hispanic'])]
Race_AgeCorr=Race_AgeCorr[['Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
print(Race_AgeCorr)
print("Counties in which older age populations are there in more percentages have lower number of blacks, asians and hispanics, i.e. they are negatively correlated to the higher ages")
ax = sns.heatmap(Race_AgeCorr.transpose(),cmap="YlGnBu" )

##############################################################################
#poverty among states mean of counties
PovertyDf=df_Demog[['Poverty']].groupby(df_Demog['CHSI_State_Name']).mean().sort_values(by=['Poverty'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
chart = sns.barplot(x=PovertyDf.index, y="Poverty", data=PovertyDf)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
################################
    
#most youngsters in usa, sum of counties
Age_19_Under_df=df_Demog[['Age_19_Under']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_19_Under'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
ax6 = plt.subplot2grid((2, 2), (0, 0))
Lowest=Age_19_Under_df.head(10)
Highest=Age_19_Under_df.tail(10)
chart = sns.barplot(x=Lowest.index, y="Age_19_Under", data=Lowest)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
chart.set_title("Top 10 States with least population of under 19", fontsize=20)

Age_19_64_df=df_Demog[['Age_19_64']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_19_64'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
chart = sns.barplot(x=Age_19_64_df.index, y="Age_19_64", data=Age_19_64_df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

Age_65_84_df=df_Demog[['Age_65_84']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_65_84'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
chart = sns.barplot(x=Age_65_84_df.index, y="Age_65_84", data=Age_65_84_df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

Age_85_and_Over_df=df_Demog[['Age_85_and_Over']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_85_and_Over'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
chart = sns.barplot(x=Age_85_and_Over_df.index, y="Age_85_and_Over", data=Age_85_and_Over_df)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

#####################################################################################################
PopSize=df_Demog[['Population_Size']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Population_Size'])
chart = sns.barplot(x=PopSize.index, y="Population_Size", data=PopSize)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
#######################################################################
Races_df_grp=df_Demog[['White','Black','Asian']].groupby(df_Demog['CHSI_State_Name']).mean().sort_values(by=['White','Black','Asian'])
Races_df_grp.head(10)
Races_df_grp.tail(10)

#########temp######################################

temp_df=df_Demog.groupby(df_Demog['CHSI_State_Name']).mean().sort_values(by=['Poverty'])
temp_df.columns

chart = sns.scatterplot(x="Population_Density", y="Poverty", data=temp_df)
text=temp_df.index

eucs=temp_df['Population_Density']#x
covers=temp_df['Poverty']

texts = []
for x, y, s in zip(eucs, covers, text):
    texts.append(plt.text(x, y, s))
adjust_text(texts, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))



