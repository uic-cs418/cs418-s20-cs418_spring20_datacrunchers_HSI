# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:17:35 2020

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
###########changing working directory################################
mypath=r'D:\Drive\Coursework\IDS\Project_IDS\Dataset'
os.chdir(mypath)
####################################################################
##run vunerable pops env health py
Demo_VPEH_df.columns
Demo_VPEH_df_tojoin=Demo_VPEH_df[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name_x',
       'CHSI_State_Name_x', 'CHSI_State_Abbr_x', 'Strata_ID_Number_x',
       'Population_Size', 'Population_Density', 'Poverty', 'Age_19_Under',
       'Age_19_64', 'Age_65_84', 'Age_85_and_Over', 'White', 'Black',
       'Native_American', 'Asian', 'Hispanic','No_HS_Diploma', 'Unemployed', 
       'Sev_Work_Disabled', 'Major_Depression',
       'Recent_Drug_Use', 'Ecol_Rpt',  'Salm_Rpt', 'Shig_Rpt','Toxic_Chem', 'No_HS_Diploma%', 'Unemployed%',
       'Sev_Work_Disabled%', 'Major_Depression%', 'Recent_Drug_Use%',
       'Poverty_log']]
Demo_VPEH_df_tojoin
#################################################
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df_PSU=pd.read_csv('PREVENTIVESERVICESUSE.csv')
Useful=['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name',
       'CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number', 'FluB_Rpt', 'HepA_Rpt','HepB_Rpt','Meas_Rpt',  'Pert_Rpt','CRS_Rpt', 'Syphilis_Rpt',
       'Pap_Smear','Mammogram', 'Proctoscopy',  'Pneumo_Vax', 'Flu_Vac']
df_PSU=df_PSU[Useful]
HandleNanCols=['FluB_Rpt',
       'HepA_Rpt', 'HepB_Rpt', 'Meas_Rpt', 'Pert_Rpt', 'CRS_Rpt',
       'Syphilis_Rpt', 'Pap_Smear', 'Mammogram', 'Proctoscopy', 'Pneumo_Vax',
       'Flu_Vac']
df_PSU[df_PSU[HandleNanCols]<0]=np.nan
#########################################################
Demo_VPEH_df_tojoin.columns
PSU_Demo_VPEH_df=df_PSU.merge(Demo_VPEH_df_tojoin, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
PSU_Demo_VPEH_df.columns
for i in range(0,len(HandleNanCols)):
    stringname=HandleNanCols[i]
    stringnameo=stringname+'%'
    PSU_Demo_VPEH_df[stringnameo]=PSU_Demo_VPEH_df[stringname]/PSU_Demo_VPEH_df['Population_Size']

def CorrelationTable(ToCorr):
    CorrelationTable=ToCorr.corr()
    CorrelationTable=CorrelationTable[(CorrelationTable>0.5) | (CorrelationTable<-0.5)]
    CorrelationTable=CorrelationTable.reset_index()
    CorrelationTable=pd.melt(CorrelationTable, id_vars=['index'])
    CorrelationTable=CorrelationTable.dropna()
    CorrelationTable=CorrelationTable[CorrelationTable['value']!=1]
    return(CorrelationTable)

Table=CorrelationTable(PSU_Demo_VPEH_df)
PSU_Demo_VPEH_df.columns
Table

#########################################################
Histogramsdf=PSU_Demo_VPEH_df.copy()
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

ax1 = plt.subplot2grid((3, 3), (0, 0))
ax1=sns.distplot((Histogramsdf['Flu_Vac'].dropna()))
ax1.set_title("Flu Vaccine")

ax2 = plt.subplot2grid((3, 3), (0, 1))
ax2=sns.distplot((Histogramsdf['Pneumo_Vax'].dropna()))
ax2.set_title("Pneumonia vaccine for 65 above ages")

ax3 = plt.subplot2grid((3, 3), (1, 0))
ax3=sns.distplot((Histogramsdf['Proctoscopy'].dropna()))
ax3.set_title("# Proctoscopy")
ax3.text(0, 0.04, 'Proctoscopy is a procedure \n used to diagnose problems\n w rectum and anus', style='italic')

ax4 = plt.subplot2grid((3, 3), (1, 1))
ax4=sns.distplot((Histogramsdf['Mammogram'].dropna()))
ax4.set_title("#Mammography")
ax4.text(56, 0.07, 'goal of mammography is the \n early detection of breast cancer', style='italic')

ax5 = plt.subplot2grid((3, 3), (2, 0))
ax5=sns.distplot((Histogramsdf['Pap_Smear'].dropna()))
ax5.set_title("# Pap Smear")
ax5.text(56, 0.07, 'cervical screening used to detect \npotentially precancerous and cancerous \n processes in the cervix or colon', style='italic')
plt.tight_layout(pad=2, w_pad=2, h_pad=2.0)
####################################################################################
####################################################################################
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

ax6 = plt.subplot2grid((3, 3), (0, 0))
ax6=sns.distplot(np.log(Histogramsdf[Histogramsdf['Syphilis_Rpt']>0]['Syphilis_Rpt']))
ax6.set_title("Syphilis reported cases")
ax6.text(0.5, 0.7, 'bacterial infection usually spread\n by sexual contact', style='italic')

ax7 = plt.subplot2grid((3, 3), (0, 1), colspan=1)
ax7=sns.distplot(np.log(Histogramsdf[Histogramsdf['HepA_Rpt']>0]['HepA_Rpt']))
ax7.set_title("Hepatitis A reported cases")
ax7.text(0.7, 0.59, 'It spreads from contaminated food or water,\n or contact with someone who is infected', style='italic')

ax8 = plt.subplot2grid((3, 3), (1, 0), colspan=1)
ax8=sns.distplot(np.log(Histogramsdf[Histogramsdf['Pert_Rpt']>0]['Pert_Rpt']))
ax8.set_title("Pertussis reported cases")
ax8.text(0.5, 0.7, 'Pertussis, also known as whooping cough,\n is a highly contagious respiratory disease.\ncaused by bacterium Bordetella pertussis.', style='italic')

ax9 = plt.subplot2grid((3, 3), (1, 1), colspan=1)
ax9=sns.distplot(np.log(Histogramsdf[Histogramsdf['Meas_Rpt']>0]['Meas_Rpt']))
ax9.set_title("Measles reported cases")
ax9.text(1, 2, 'Measles is a highly contagious \ninfectious disease caused by \n measles virus', style='italic')

ax10 = plt.subplot2grid((3, 3), (2, 0), colspan=1)
ax10=sns.distplot(np.log(Histogramsdf[Histogramsdf['HepB_Rpt']>0]['HepB_Rpt']))
ax10.set_title("Hepatitis B reported cases")
ax10.text(0.4, 0.9, 'This disease is most commonly spread \nby exposure to infected body fluids.', style='italic')

#
ax11 = plt.subplot2grid((3, 3), (2, 1), colspan=1)
ax11=sns.distplot(np.log(Histogramsdf[Histogramsdf['FluB_Rpt']>0]['FluB_Rpt']))
ax11.set_title("Haemophilus Influenzae B  reported cases")
ax11.text(0.8, 1.5, 'infection caused by bacteria', style='italic')

plt.tight_layout(pad=2, w_pad=2, h_pad=2.0)

########################################################33
Plot1=sns.scatterplot(x='Major_Depression', y='Recent_Drug_Use', data=PSU_Demo_VPEH_df)
Plot1.set_yscale('log')
Plot1.set_xscale('log')
print("Strong Correlation of Depression & Drug Usage Causation or Correlation?")

ax = sns.scatterplot(x="Recent_Drug_Use", y="HepA_Rpt", data=PSU_Demo_VPEH_df)
ax.set_xscale('log')
ax.set_yscale('log')

ax = sns.scatterplot(x="Recent_Drug_Use", y="HepB_Rpt", data=PSU_Demo_VPEH_df)
ax.set_xscale('log')
ax.set_yscale('log')

ax = sns.scatterplot(x="Recent_Drug_Use", y="FluB_Rpt", data=PSU_Demo_VPEH_df)

ax = sns.scatterplot(x="Recent_Drug_Use", y="Syphilis_Rpt", data=PSU_Demo_VPEH_df)
ax.set_yscale('log')
ax.set_xscale('log')

PSU_Demo_VPEH_df.columns

Table=Table.reset_index()
Table['new']='['+Table['index']+', '+Table['variable']+']'