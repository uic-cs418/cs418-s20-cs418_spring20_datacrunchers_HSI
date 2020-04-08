# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:34:58 2020

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
RFAC=pd.read_csv('RISKFACTORSANDACCESSTOCARE.csv')
RFAC.columns
RFAC=RFAC[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name','CHSI_State_Name', 'CHSI_State_Abbr'
           , 'Strata_ID_Number', 'No_Exercise', 'Few_Fruit_Veg','Obesity', 'High_Blood_Pres',
           'Smoker', 'Diabetes', 'Uninsured','Elderly_Medicare', 'Disabled_Medicare', 
           'Prim_Care_Phys_Rate','Dentist_Rate']]
HandleNanCols=['No_Exercise', 'Few_Fruit_Veg','Obesity', 'High_Blood_Pres',
           'Smoker', 'Diabetes', 'Uninsured','Elderly_Medicare', 'Disabled_Medicare', 
           'Prim_Care_Phys_Rate','Dentist_Rate']
RFAC[RFAC[HandleNanCols]<0]=np.nan

PSU_Demo_VPEH_SMOH_df= PSU_Demo_VPEH_SMOH_df[['State_FIPS_Code', 'County_FIPS_Code',
       'FluB_Rpt', 'HepA_Rpt', 'HepB_Rpt', 'Meas_Rpt', 'Pert_Rpt', 'CRS_Rpt',
       'Syphilis_Rpt', 'Pap_Smear', 'Mammogram', 'Proctoscopy', 'Pneumo_Vax',
       'Flu_Vac', 'Population_Size', 'Population_Density', 'Poverty',
       'Age_19_Under', 'Age_19_64', 'Age_65_84', 'Age_85_and_Over', 'White',
       'Black', 'Native_American', 'Asian', 'Hispanic', 'No_HS_Diploma',
       'Unemployed', 'Sev_Work_Disabled', 'Major_Depression',
       'Recent_Drug_Use', 'Ecol_Rpt', 'Salm_Rpt', 'Shig_Rpt', 'Toxic_Chem',
       'No_HS_Diploma%', 'Unemployed%', 'Sev_Work_Disabled%',
       'Major_Depression%', 'Recent_Drug_Use%', 'FluB_Rpt%', 'HepA_Rpt%',
       'HepB_Rpt%', 'Meas_Rpt%', 'Pert_Rpt%', 'CRS_Rpt%', 'Syphilis_Rpt%',
       'Pap_Smear%', 'Mammogram%', 'Proctoscopy%', 'Pneumo_Vax%', 'Flu_Vac%',
       'ALE', 'All_Death', 'Health_Status',
       'Unhealthy_Days']]
PSU_Demo_VPEH_SMOH_RFAC_df=RFAC.merge(PSU_Demo_VPEH_SMOH_df, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
PSU_Demo_VPEH_SMOH_RFAC_df.columns

iris=PSU_Demo_VPEH_SMOH_RFAC_df[['No_Exercise','Obesity', 'High_Blood_Pres',
           'Smoker', 'Diabetes']]
g = sns.PairGrid(iris)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=15)

