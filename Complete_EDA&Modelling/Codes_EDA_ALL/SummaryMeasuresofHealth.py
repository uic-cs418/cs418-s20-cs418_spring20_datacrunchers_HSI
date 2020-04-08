# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:51:56 2020

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
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
####################DEMOGRAPHICS#########################################
##descriptive statistics
df_SMOH=pd.read_csv('SUMMARYMEASURESOFHEALTH.csv')
ForUse=['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name','CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number']
UsefulCols=['ALE', 'All_Death', 'Health_Status', 'Unhealthy_Days']
df_SMOH[df_SMOH[UsefulCols]<0]=np.nan
df_SMOH=df_SMOH[ForUse+UsefulCols]
PSU_Demo_VPEH_df=PSU_Demo_VPEH_df[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name', 'CHSI_State_Name',
 'CHSI_State_Abbr', 'Strata_ID_Number', 'FluB_Rpt', 'HepA_Rpt', 'HepB_Rpt', 
 'Meas_Rpt', 'Pert_Rpt', 'CRS_Rpt',      
 'Syphilis_Rpt', 'Pap_Smear', 'Mammogram', 'Proctoscopy', 'Pneumo_Vax',
 'Flu_Vac', 'Population_Size',
 'Population_Density', 'Poverty', 'Age_19_Under', 'Age_19_64',
 'Age_65_84', 'Age_85_and_Over', 'White', 'Black', 'Native_American',
 'Asian', 'Hispanic', 'No_HS_Diploma', 'Unemployed', 'Sev_Work_Disabled',
 'Major_Depression', 'Recent_Drug_Use', 'Ecol_Rpt', 'Salm_Rpt',
 'Shig_Rpt', 'Toxic_Chem', 'No_HS_Diploma%', 'Unemployed%',
 'Sev_Work_Disabled%', 'Major_Depression%', 'Recent_Drug_Use%',
 'FluB_Rpt%', 'HepA_Rpt%', 'HepB_Rpt%',
 'Meas_Rpt%', 'Pert_Rpt%', 'CRS_Rpt%', 'Syphilis_Rpt%', 'Pap_Smear%',
 'Mammogram%', 'Proctoscopy%', 'Pneumo_Vax%', 'Flu_Vac%']]
df_SMOH.columns
PSU_Demo_VPEH_SMOH_df=PSU_Demo_VPEH_df.merge(df_SMOH, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
PSU_Demo_VPEH_SMOH_df.columns
###################plots########################################################
sns.lmplot('Poverty', 'All_Death', data=PSU_Demo_VPEH_SMOH_df, ci=None, order=2, truncate=True, palette="Set1")
sns.lmplot('Poverty', 'ALE', data=PSU_Demo_VPEH_SMOH_df, ci=None, order=2, truncate=True, palette="Set1")


