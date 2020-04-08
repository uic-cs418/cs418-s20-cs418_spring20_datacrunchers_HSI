# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:13:47 2020

@author: Varun
"""

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
df_VPEH=pd.read_csv('VUNERABLEPOPSANDENVHEALTH.csv')
df_VPEH.columns
#run demographics.py
Demo_VPEH_df=df_Demog.merge(df_VPEH, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
Demo_VPEH_df.columns
Demo_VPEH_df=Demo_VPEH_df.replace([i for i in ListofNans], np.NAN)
#####################################################################################################
imp_cols=['No_HS_Diploma', 'Unemployed', 'Sev_Work_Disabled', 'Major_Depression','Recent_Drug_Use']
Demo_VPEH_df[imp_cols[0]+str('%')]=Demo_VPEH_df[imp_cols[0]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[1]+str('%')]=Demo_VPEH_df[imp_cols[1]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[2]+str('%')]=Demo_VPEH_df[imp_cols[2]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[3]+str('%')]=Demo_VPEH_df[imp_cols[3]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[4]+str('%')]=Demo_VPEH_df[imp_cols[4]]/Demo_VPEH_df['Population_Size']

plot_cols=['No_HS_Diploma%', 'Unemployed%','Sev_Work_Disabled%', 'Major_Depression%', 'Recent_Drug_Use%']
#run for all is
i=3
plot1=Demo_VPEH_df[plot_cols].groupby(Demo_VPEH_df['CHSI_State_Name_x']).mean().sort_values(by=plot_cols[i])
sns.set(rc={'figure.figsize':(11.7,8.27)})
chart = sns.barplot(x=plot1.index, y=plot1[plot_cols[i]], data=plot1)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
################################################################################################]
#Grouped_Demo_VPEH_df=Demo_VPEH_df.groupby(Demo_VPEH_df['CHSI_State_Name_x']).mean()
Demo_VPEH_df['Poverty_log']=np.log(Demo_VPEH_df['Poverty'])
Demo_VPEH_df['Unemployed%_log']=np.log(Demo_VPEH_df['Unemployed%'])
Demo_VPEH_df['Major_Depression%_log']=np.log(Demo_VPEH_df['Major_Depression%'])
Demo_VPEH_df['Major_Depression_log']=np.log(Demo_VPEH_df['Major_Depression'])
Demo_VPEH_df['Population_Density_log']=np.log(Demo_VPEH_df['Population_Density'])
Demo_VPEH_df['Toxic_Chem_log']=np.log(Demo_VPEH_df['Toxic_Chem'])
Demo_VPEH_df['Population_Size_log']=np.log(Demo_VPEH_df['Population_Size'])
Demo_VPEH_df['Ecol_Rpt_log']=np.log(Demo_VPEH_df['Ecol_Rpt'])
Demo_VPEH_df['Ecol_Salm_Shig']=Demo_VPEH_df['Ecol_Rpt']+Demo_VPEH_df['Salm_Rpt']+Demo_VPEH_df['Shig_Rpt']




##Poverty & Unemployment
g =sns.FacetGrid(Demo_VPEH_df, col='CHSI_State_Name_x',col_wrap=5)
g =(g.map(plt.scatter, "Poverty", "Unemployed%_log", edgecolor="w").add_legend())
print("there is a positive correlation of poverty and unemployed as expected in most states")

##Poverty & PopulationDensity
g =sns.FacetGrid(Demo_VPEH_df, col='CHSI_State_Name_x',col_wrap=5)
g =(g.map(plt.scatter, "Poverty", 'Population_Density_log').add_legend())
print("more dense areas are seemingly with less poverty, maybe because more job opportunities?")

##Poverty & Major Depression
g =sns.FacetGrid(Demo_VPEH_df, col='CHSI_State_Name_x',col_wrap=5)
g =(g.map(plt.scatter, "Major_Depression_log", 'Poverty').add_legend())
print("why no relation between depression and poverty?")

##Major Depression & Population Density
g =sns.FacetGrid(Demo_VPEH_df, col='CHSI_State_Name_x',col_wrap=5)
g =(g.map(plt.scatter, 'Major_Depression_log', 'Population_Density_log').add_legend())
print("strong correlation of major depression and pop density")

###Poverty & No High School Diploma Numbers
g =sns.FacetGrid(Demo_VPEH_df, col='CHSI_State_Name_x',col_wrap=5)
g =(g.map(plt.scatter, 'Poverty', 'No_HS_Diploma%').add_legend())
print("positive correlation between poverty and no hs diploma")


###most correlated columns
ToCorr=Demo_VPEH_df[['Population_Size', 'Population_Density', 'Poverty', 'Age_19_Under',
       'Age_19_64', 'Age_65_84', 'Age_85_and_Over', 'White', 'Black',
       'Native_American', 'Asian', 'Hispanic','No_HS_Diploma%', 'Unemployed%',
       'Sev_Work_Disabled%', 'Major_Depression%', 'Recent_Drug_Use%',
       'Ecol_Rpt', 'Ecol_Rpt_Ind', 'Ecol_Exp', 'Salm_Rpt',
       'Salm_Rpt_Ind', 'Salm_Exp', 'Shig_Rpt', 'Shig_Rpt_Ind', 'Shig_Exp',
       'Toxic_Chem']]

CorrelationTable=ToCorr.corr()
CorrelationTable=CorrelationTable[(CorrelationTable>0.5) | (CorrelationTable<-0.5)]
CorrelationTable=CorrelationTable.reset_index()
CorrelationTable=pd.melt(CorrelationTable, id_vars=['index'])
CorrelationTable=CorrelationTable.dropna()
CorrelationTable=CorrelationTable[CorrelationTable['value']!=1]

"""
1.NO High school diploma correlated with poverty
2. Population size and E.Colli, Salmonella and Shigella Correlated
these are hygiene related diseases spread through bad food/human/animal feces

"""
Demo_VPEH_dfTemp=Demo_VPEH_df[Demo_VPEH_df['Ecol_Salm_Shig']>100]
Demo_VPEH_dfTemp['Ecol_Salm_Shig']=np.log(Demo_VPEH_dfTemp['Ecol_Salm_Shig'])

Plot1=sns.scatterplot(x='No_HS_Diploma%', y='Poverty', data=Demo_VPEH_df)
Plot2=sns.scatterplot(x='Population_Size_log', y='Ecol_Salm_Shig', data=Demo_VPEH_dfTemp)

##race vs poverty
Demo_VPEH_dfTemp['Asian']=np.log(Demo_VPEH_dfTemp['Asian'])
Plot3=sns.scatterplot(x='Poverty', y='White', data=Demo_VPEH_df)
Plot4=sns.scatterplot(x='Poverty', y='Asian', data=Demo_VPEH_dfTemp)
Plot5=sns.scatterplot(x='Poverty', y='Black', data=Demo_VPEH_dfTemp)
