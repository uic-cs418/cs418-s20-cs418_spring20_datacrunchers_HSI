# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:44:30 2020

@author: Varun
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:53:24 2020

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

pd.set_option("display.max_rows", None)
DD=pd.read_csv('DATAELEMENTDESCRIPTION.csv')
DD=DD[['PAGE_NAME','COLUMN_NAME','DESCRIPTION','IS_PERCENT_DATA']]
ColstoUse=pd.read_csv('ColstoUse.csv')
DD=DD[DD['COLUMN_NAME'].isin(list(ColstoUse['Cols']))]
df_Demog=pd.read_csv('DEMOGRAPHICS.csv')
df_Demog=df_Demog[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name','CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number','Population_Size','Population_Density','Poverty','Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over','White','Black', 'Native_American','Asian', 'Hispanic']]
ListofNans=[-9999,-2222,-2222.2,-2,-1111.1,-1111,-1,-9998.9]
df_Demog=df_Demog.replace([i for i in ListofNans], np.NAN)#replacing odd values with nan

PovertyStats=df_Demog['Poverty'].describe()
Races_df=df_Demog[['White','Black','Asian', 'Hispanic']]
Ages_df=df_Demog[['Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
Race_Age=df_Demog[['White','Black','Asian', 'Hispanic','Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
Race_AgeCorr=pd.DataFrame(Race_Age.corr())
Race_AgeCorr=Race_AgeCorr[Race_AgeCorr.index.isin(['White','Black','Asian', 'Hispanic'])]
Race_AgeCorr=Race_AgeCorr[['Age_19_Under','Age_19_64', 'Age_65_84','Age_85_and_Over']]
PovertyDf=df_Demog[['Poverty']].groupby(df_Demog['CHSI_State_Name']).mean().sort_values(by=['Poverty'])
Age_19_Under_df=df_Demog[['Age_19_Under']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_19_Under'])
Age_19_64_df=df_Demog[['Age_19_64']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_19_64'])
Age_65_84_df=df_Demog[['Age_65_84']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_65_84'])
Age_85_and_Over_df=df_Demog[['Age_85_and_Over']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Age_85_and_Over'])
PopSize=df_Demog[['Population_Size']].groupby(df_Demog['CHSI_State_Name']).sum().sort_values(by=['Population_Size'])
Races_df_grp=df_Demog[['White','Black','Asian']].groupby(df_Demog['CHSI_State_Name']).mean().sort_values(by=['White','Black','Asian'])
df_VPEH=pd.read_csv('VUNERABLEPOPSANDENVHEALTH.csv')
Demo_VPEH_df=df_Demog.merge(df_VPEH, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
Demo_VPEH_df=Demo_VPEH_df.replace([i for i in ListofNans], np.NAN)
imp_cols=['No_HS_Diploma', 'Unemployed', 'Sev_Work_Disabled', 'Major_Depression','Recent_Drug_Use']
Demo_VPEH_df[imp_cols[0]+str('%')]=Demo_VPEH_df[imp_cols[0]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[1]+str('%')]=Demo_VPEH_df[imp_cols[1]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[2]+str('%')]=Demo_VPEH_df[imp_cols[2]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[3]+str('%')]=Demo_VPEH_df[imp_cols[3]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df[imp_cols[4]+str('%')]=Demo_VPEH_df[imp_cols[4]]/Demo_VPEH_df['Population_Size']
Demo_VPEH_df['Poverty_log']=np.log(Demo_VPEH_df['Poverty'])
Demo_VPEH_df['Unemployed%_log']=np.log(Demo_VPEH_df['Unemployed%'])
Demo_VPEH_df['Major_Depression%_log']=np.log(Demo_VPEH_df['Major_Depression%'])
Demo_VPEH_df['Major_Depression_log']=np.log(Demo_VPEH_df['Major_Depression'])
Demo_VPEH_df['Population_Density_log']=np.log(Demo_VPEH_df['Population_Density'])
Demo_VPEH_df['Toxic_Chem_log']=np.log(Demo_VPEH_df['Toxic_Chem'])
Demo_VPEH_df['Population_Size_log']=np.log(Demo_VPEH_df['Population_Size'])
Demo_VPEH_df['Ecol_Rpt_log']=np.log(Demo_VPEH_df['Ecol_Rpt'])
Demo_VPEH_df['Ecol_Salm_Shig']=Demo_VPEH_df['Ecol_Rpt']+Demo_VPEH_df['Salm_Rpt']+Demo_VPEH_df['Shig_Rpt']
Demo_VPEH_dfTemp=Demo_VPEH_df[Demo_VPEH_df['Ecol_Salm_Shig']>100]
"""
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
Demo_VPEH_dfTemp['Ecol_Salm_Shig']=np.log(Demo_VPEH_dfTemp['Ecol_Salm_Shig'])
Plot1=sns.scatterplot(x='No_HS_Diploma%', y='Poverty', data=Demo_VPEH_dfTemp)
plt.title('No High School Diploma Percentages & Poverty')

Plot2=sns.scatterplot(x='Population_Size_log', y='Ecol_Salm_Shig', data=Demo_VPEH_dfTemp)
plt.title('Population Size Vs Cases of Hygiene Related Diseases')
plt.rcParams["figure.figsize"] = (5,5)
    
plt.rcParams["figure.figsize"] = (8,8)
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1=sns.scatterplot(x='Poverty', y='Black', data=Demo_VPEH_dfTemp)
plt.title('Poverty & Race')

ax2 = fig.add_subplot(gs[1, 0])
Demo_VPEH_dfTemp['Asian_log']=np.log(Demo_VPEH_dfTemp['Asian'])
ax2=sns.scatterplot(x='Poverty', y='Asian_log', data=Demo_VPEH_dfTemp)
#plt.title('Poverty & Race: Asian')

ax3 = fig.add_subplot(gs[0, 1])
ax3=sns.scatterplot(x='Poverty', y='White', data=Demo_VPEH_df)
plt.title('Poverty & Race')

########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################"""
Demo_VPEH_df_tojoin=Demo_VPEH_df[['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name_x',
       'CHSI_State_Name_x', 'CHSI_State_Abbr_x', 'Strata_ID_Number_x',
       'Population_Size', 'Population_Density', 'Poverty', 'Age_19_Under',
       'Age_19_64', 'Age_65_84', 'Age_85_and_Over', 'White', 'Black',
       'Native_American', 'Asian', 'Hispanic','No_HS_Diploma', 'Unemployed', 
       'Sev_Work_Disabled', 'Major_Depression',
       'Recent_Drug_Use', 'Ecol_Rpt',  'Salm_Rpt', 'Shig_Rpt','Toxic_Chem', 'No_HS_Diploma%', 'Unemployed%',
       'Sev_Work_Disabled%', 'Major_Depression%', 'Recent_Drug_Use%',
       'Poverty_log']]
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
PSU_Demo_VPEH_df=df_PSU.merge(Demo_VPEH_df_tojoin, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
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
"""
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
Plot1=sns.scatterplot(x='Major_Depression', y='Recent_Drug_Use', data=PSU_Demo_VPEH_df)
Plot1.set_yscale('log')
Plot1.set_xscale('log')
print("Strong Correlation of Depression & Drug Usage Causation or Correlation?")
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################"""

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

"""
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
ax1=sns.lmplot('Poverty', 'All_Death', data=PSU_Demo_VPEH_SMOH_df, ci=None, order=2, truncate=True, palette="Set1")
fig = ax1.fig 
fig.suptitle("Poverty & Number of Deaths in a county are positively correlated", fontsize=15)

ax2=sns.lmplot('Poverty', 'ALE', data=PSU_Demo_VPEH_SMOH_df, ci=None, order=2, truncate=True, palette="Set1")
fig = ax2.fig 
fig.suptitle("Poverty & Average Life Expectancy in a county are negatively correlated", fontsize=15)
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################

"""
RFAC=pd.read_csv('RISKFACTORSANDACCESSTOCARE.csv')
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
"""
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
g = sns.PairGrid(iris)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=15)
"""
MOBAD=pd.read_csv('MEASURESOFBIRTHANDDEATH.csv')
MOBAD.columns
UsefulCols=[
"LBW",
"VLBW",
"Premature",
"Under_18",
"Over_40",
"Unmarried",
"Late_Care",
"Infant_Mortality",
"IM_Neonatal",
"IM_Postneonatal",
"Brst_Cancer",
"Col_Cancer",
"CHD",
"Homicide",
"Lung_Cancer",
"MVA",
"Stroke",
"Suicide",
"Injury",
"Total_Births",
"Total_Deaths"]
ForUse=['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name','CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number']
MOBAD[MOBAD[UsefulCols]<0]=np.nan
MOBAD=MOBAD[ForUse+UsefulCols]
PSU_Demo_VPEH_SMOH_RFAC_df=PSU_Demo_VPEH_SMOH_RFAC_df.drop(columns=['_merge'])
PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df=PSU_Demo_VPEH_SMOH_RFAC_df.merge(MOBAD, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
PCT=["Infant_Mortality",
"IM_Neonatal",
"IM_Postneonatal",
"Brst_Cancer",
"Col_Cancer",
"CHD",
"Homicide",
"Lung_Cancer",
"MVA",
"Stroke",
"Suicide",
"Injury",
"Total_Births",
"Total_Deaths"]
i=0
for i in range(0,len(PCT)):
    name=PCT[i]+"%"
    PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df[name]=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df[PCT[i]]/PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df['Population_Size']
"""    
########################plots for report#####################################
########################plots for report#####################################
########################plots for report#####################################
sns.set(rc={'figure.figsize':(10,10)})
ax=sns.scatterplot('Total_Deaths', 'Uninsured', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Total Number of Deaths & Number of People UnInsured are Correlated (Log Scale)")


sns.set(rc={'figure.figsize':(10,10)})
ax=sns.scatterplot('Obesity', 'CHD', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
#ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_title("Obesity Vs Number of People who got coronory heart disease - Positive Correlation")
"""

reqcols=[
        'No_Exercise',
'Few_Fruit_Veg',
'Obesity',
'High_Blood_Pres',
'Smoker',
'Diabetes',
'Uninsured',
'Prim_Care_Phys_Rate',
'Population_Density',
'Poverty',
'No_HS_Diploma',
'Unemployed',
'Sev_Work_Disabled',
'Major_Depression',
'Recent_Drug_Use',
'Premature',
'Unmarried',
'Brst_Cancer',
'Col_Cancer',
'CHD',
'Homicide',
'Lung_Cancer',
'MVA',
'Stroke',
'Suicide',
'ALE']

heat=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df[reqcols]
plt.rcParams["figure.figsize"] = (40,18)
sns.set(font_scale=1.5)
ax = sns.heatmap(heat.corr(),annot=True,linewidths=.2)
ax.set_title('Correlation Values: What is correlated with what (on a county level)', fontsize=30)
plt.text(1,35,'Few Observations:\nCounties with more "No High School Diploma Percentages", has high correlation with drug use rate, depression rate and unemployment rate.\nAverage life expectancy of a county is negatively correlated with obesity and high blood pressure.\nNo Excercise Levels of a county are positively Correlated with Obesity and high Blood Pressure, Heart Disease.\nLung Cancer & Smoker are highly positively correlated',fontsize=30)

"""
mask = np.zeros_like(heat.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(26, 12))
    ax = sns.heatmap(heat.corr(), mask=mask, annot=True)"""
"""
g = sns.PairGrid(iris)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_offdiag(plt.scatter, edgecolor="w", s=15)"""