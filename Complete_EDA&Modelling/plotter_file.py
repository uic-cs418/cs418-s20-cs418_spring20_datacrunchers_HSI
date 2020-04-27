#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


myFilePath = r'C:\Users\apoor\Downloads\chsi_dataset'
os.chdir(myFilePath)
files = [file for file in listdir(myFilePath) if isfile(join(myFilePath,file))]
files


# In[3]:


df_leadDeathCauses = pd.read_csv('LEADINGCAUSESOFDEATH.csv')
NanList = [-9999,-2222,-2222.2,-2,-1111,-1111.1,-9998.9]


# In[4]:


#Major Depression v/s Suicide
vulnerable_pop_and_env_health_df = pd.read_csv('VunerablePopsAndEnvHealth.csv') 
vulnerable_pop_and_env_health_df.head()


# In[5]:


#Suicide Dataframe 
df_Suicide_lead_cause = df_leadDeathCauses[['County_FIPS_Code','CHSI_State_Name','CHSI_County_Name','C_Wh_Suicide','C_Bl_Suicide',
                                                          'C_Ot_Suicide','C_Hi_Suicide','D_Wh_Suicide','D_Bl_Suicide','D_Ot_Suicide','D_Hi_Suicide']]
df_Suicide_lead_cause = df_Suicide_lead_cause.replace([i for i in NanList],np.NAN)
df_Suicide_lead_cause


# In[6]:


Demo_df = pd.read_csv('DEMOGRAPHICS.csv')
Demo_df_populationSize = Demo_df[['County_FIPS_Code','CHSI_County_Name','CHSI_State_Name','White','Black',
                                  'Hispanic','Native_American','Asian','Population_Size']]
Demo_df_populationSize = Demo_df_populationSize.replace([i for i in NanList],np.NAN)
#print(Demo_df_populationSize.head())
white_population_percentage = Demo_df_populationSize['White']
Total_white_population_arr = [(x*y)/100 for x,y in zip(white_population_percentage,Demo_df_populationSize['Population_Size'])]

black_population_percentage = Demo_df_populationSize['Black']
Total_black_population_arr = [(x*y)/100 for x,y in zip(black_population_percentage,Demo_df_populationSize['Population_Size'])]

hisp_population_percentage = Demo_df_populationSize['Hispanic']
Total_Hispanic_population_arr = [(x*y)/100 for x,y in zip(hisp_population_percentage,Demo_df_populationSize['Population_Size'])]

NativeAmer_population_percentage = Demo_df_populationSize['Native_American']
Asian_population_percentage = Demo_df_populationSize['Asian']

OtherPopulation_percentage = [(x+y) for x,y in zip(Asian_population_percentage,NativeAmer_population_percentage)]
Other_population_arr = [(x*y)/100 for x,y in zip(OtherPopulation_percentage,Demo_df_populationSize['Population_Size'])]

Demo_df_populationSize['White Population'] = pd.Series(Total_white_population_arr,index=Demo_df_populationSize.index)
Demo_df_populationSize['Black Population'] = pd.Series(Total_black_population_arr,index=Demo_df_populationSize.index)
Demo_df_populationSize['Hispanic Population'] = pd.Series(Total_Hispanic_population_arr,index=Demo_df_populationSize.index)
Demo_df_populationSize['Other Population'] = pd.Series(Other_population_arr,index=Demo_df_populationSize.index)
Demo_df_populationSize


# In[13]:


df_riskfactors = pd.read_csv('RISKFACTORSANDACCESSTOCARE.csv')
df_riskFactorSmoker = df_riskfactors[['County_FIPS_Code','Smoker','CHSI_County_Name','CHSI_State_Name']]
df_riskFactorSmoker = df_riskFactorSmoker.replace([i for i in NanList],np.NAN)
merged_df_vul_suicide = pd.merge(df_Suicide_lead_cause,vulnerable_pop_and_env_health_df['Major_Depression'],how='left',on=df_Suicide_lead_cause.index)
del merged_df_vul_suicide['key_0']
merged_df_vul_suicide = pd.merge(merged_df_vul_suicide,Demo_df_populationSize,how='left',on=merged_df_vul_suicide.index)
merged_df_vul_suicide.fillna(0,inplace=True)

white_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Wh_Suicide'],merged_df_vul_suicide['D_Wh_Suicide'],merged_df_vul_suicide['White Population'])]

black_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Bl_Suicide'],merged_df_vul_suicide['D_Bl_Suicide'],merged_df_vul_suicide['Black Population'])]


hispanic_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Hi_Suicide'],merged_df_vul_suicide['D_Hi_Suicide'],merged_df_vul_suicide['Hispanic Population'])]
hispanic_population_Suicide

other_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Ot_Suicide'],merged_df_vul_suicide['D_Ot_Suicide'],merged_df_vul_suicide['Other Population'])]
other_population_Suicide

merged_df_vul_suicide['White_Population_Suicide'] = pd.Series(white_population_Suicide,index=merged_df_vul_suicide.index)
merged_df_vul_suicide['Black_Population_Suicide'] = pd.Series(black_population_Suicide,index=merged_df_vul_suicide.index)
merged_df_vul_suicide['Hispanic_Population_Suicide'] = pd.Series(hispanic_population_Suicide,index=merged_df_vul_suicide.index)
merged_df_vul_suicide['Other_Population_Suicide'] = pd.Series(other_population_Suicide,index=merged_df_vul_suicide.index)

tot_population_Suicide = [(a+b+c+d) for a,b,c,d in zip(merged_df_vul_suicide['White_Population_Suicide'],merged_df_vul_suicide['Other_Population_Suicide'],merged_df_vul_suicide['Black_Population_Suicide'],merged_df_vul_suicide['Hispanic_Population_Suicide'])]
merged_df_vul_suicide['Total_Suicide_Population'] = pd.Series(tot_population_Suicide,index = merged_df_vul_suicide.index)

corrs = (merged_df_vul_suicide[['Major_Depression', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_df_vul_suicide.Total_Suicide_Population)
        .rename(columns={'Major_Depression' : 'Corr_Coef'}))
corrs_depression = corrs[corrs['Corr_Coef'].notnull()]

#no exercise v/s Heart Disease
#High blood pressure v/s Heart Disease
df_HeartDisease_lead_cause = df_leadDeathCauses[['County_FIPS_Code','CHSI_State_Name','CHSI_County_Name','D_Wh_HeartDis','D_Bl_HeartDis',
                                                          'D_Ot_HeartDis','D_Hi_HeartDis','E_Wh_HeartDis','E_Bl_HeartDis','E_Ot_HeartDis','E_Hi_HeartDis','F_Wh_HeartDis',
                                                'F_Bl_HeartDis','F_Ot_HeartDis','F_Hi_HeartDis']]
df_HeartDisease_lead_cause = df_HeartDisease_lead_cause.replace([i for i in NanList],np.NAN)

df_riskFactorHighBP = df_riskfactors[['County_FIPS_Code','No_Exercise','CHSI_County_Name','CHSI_State_Name']]
df_riskFactorHighBP = df_riskFactorHighBP.replace([i for i in NanList],np.NAN)


merged_risk_factor_LeadCauseHeartDis = pd.merge(df_HeartDisease_lead_cause,df_riskFactorHighBP['No_Exercise'],how='left',on=df_HeartDisease_lead_cause.index)
del merged_risk_factor_LeadCauseHeartDis['key_0']
merged_demo_risk_factor_LeadCauseDeath = pd.merge(merged_risk_factor_LeadCauseHeartDis,Demo_df_populationSize,how='left',on=merged_risk_factor_LeadCauseHeartDis.index)
merged_demo_risk_factor_LeadCauseDeath.fillna(0,inplace=True)

white_population_HeartDis = [((b+c+d)*y)/100 for b,c,d,y in zip(merged_demo_risk_factor_LeadCauseDeath['D_Wh_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['E_Wh_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['F_Wh_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['White Population'])]

black_population_HeartDis = [((b+c+d)*y)/100 for b,c,d,y in zip(merged_demo_risk_factor_LeadCauseDeath['D_Bl_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['E_Bl_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['F_Bl_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Black Population'])]

hispanic_population_HeartDis = [((b+c+d)*y)/100 for b,c,d,y in zip(merged_demo_risk_factor_LeadCauseDeath['D_Hi_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['E_Hi_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['F_Hi_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Hispanic Population'])]

other_population_HeartDis = [((b+c+d)*y)/100 for b,c,d,y in zip(merged_demo_risk_factor_LeadCauseDeath['D_Ot_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['E_Ot_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['F_Ot_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Other Population'])]

merged_demo_risk_factor_LeadCauseDeath['White_Population_HeartDis'] = pd.Series(white_population_HeartDis,index=merged_demo_risk_factor_LeadCauseDeath.index)
merged_demo_risk_factor_LeadCauseDeath['Black_Population_HeartDis'] = pd.Series(black_population_HeartDis,index=merged_demo_risk_factor_LeadCauseDeath.index)
merged_demo_risk_factor_LeadCauseDeath['Hispanic_Population_HeartDis'] = pd.Series(hispanic_population_HeartDis,index=merged_demo_risk_factor_LeadCauseDeath.index)
merged_demo_risk_factor_LeadCauseDeath['Other_Population_HeartDis'] = pd.Series(other_population_HeartDis,index=merged_demo_risk_factor_LeadCauseDeath.index)


tot_population_HeartDis = [((a+b+c+d)/e)*100 for a,b,c,d,e in zip(merged_demo_risk_factor_LeadCauseDeath['White_Population_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Black_Population_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Hispanic_Population_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Other_Population_HeartDis'],merged_demo_risk_factor_LeadCauseDeath['Population_Size'])]
merged_demo_risk_factor_LeadCauseDeath['Percentage_Total_HeartDis_Population'] = pd.Series(tot_population_HeartDis,index = merged_demo_risk_factor_LeadCauseDeath.index)

corrs = (merged_demo_risk_factor_LeadCauseDeath[['No_Exercise', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_demo_risk_factor_LeadCauseDeath.Percentage_Total_HeartDis_Population)
        .rename(columns={'No_Exercise' : 'Corr_Coef'}))
corrs_NoExercise_HeartDis = corrs[corrs['Corr_Coef'].notnull()]

#Birth Defects v/s Premature deliveries
measures_birth_Death_df = pd.read_csv('MEASURESOFBIRTHANDDEATH.csv')
measures_birth_Death_df

premature_df = measures_birth_Death_df[['CHSI_County_Name','CHSI_State_Name','Premature']].copy()
premature_df = premature_df.replace([i for i in NanList],np.NAN)
premature_df



df_leadDeathCauses_bf = df_leadDeathCauses[['State_FIPS_Code','County_FIPS_Code','CHSI_County_Name','CHSI_State_Name','A_Wh_BirthDef','A_Bl_BirthDef','A_Hi_BirthDef','A_Ot_BirthDef']].copy()
NanList = [-9999,-2222,-2222.2,-2,-1111,-1111.1,-9998.9]
df_leadDeathCauses_bf = df_leadDeathCauses_bf.replace([i for i in NanList],np.NAN)
df_leadDeathCauses_bf.head()

merged_premature_LeadCauseBirthDef = pd.merge(df_leadDeathCauses_bf,premature_df['Premature'],how='left',on=df_leadDeathCauses_bf.index)
del merged_premature_LeadCauseBirthDef['key_0']
merged_demo_premature_LeadCauseBf = pd.merge(merged_premature_LeadCauseBirthDef,Demo_df_populationSize,how='left',on=merged_premature_LeadCauseBirthDef.index)
merged_demo_premature_LeadCauseBf.fillna(0,inplace=True)
merged_demo_premature_LeadCauseBf

white_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Wh_BirthDef'],merged_demo_risk_factor_LeadCauseDeath['White Population'])]

black_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Bl_BirthDef'],merged_demo_premature_LeadCauseBf['Black Population'])]

hispanic_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Hi_BirthDef'],merged_demo_premature_LeadCauseBf['Hispanic Population'])]

other_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Ot_BirthDef'],merged_demo_premature_LeadCauseBf['Other Population'])]

merged_demo_premature_LeadCauseBf['White_Population_BirthDef'] = pd.Series(white_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Black_Population_BirthDef'] = pd.Series(black_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Hispanic_Population_BirthDef'] = pd.Series(hispanic_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Other_Population_BirthDef'] = pd.Series(other_population_bf,index=merged_demo_premature_LeadCauseBf.index)


tot_population_Bf = [((a+b+c+d)/e)*100 for a,b,c,d,e in zip(merged_demo_premature_LeadCauseBf['White_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Black_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Hispanic_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Other_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Population_Size'])]
merged_demo_premature_LeadCauseBf['Percentage_Total_Bf_Population'] = pd.Series(tot_population_Bf,index = merged_demo_premature_LeadCauseBf.index)
corrs = (merged_demo_premature_LeadCauseBf[['Premature', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_demo_premature_LeadCauseBf.Percentage_Total_Bf_Population)
        .rename(columns={'Premature' : 'Corr_Coef'}))
corrs_Bf = corrs[corrs['Corr_Coef'].notnull()]

#Birth to women over 40 v/s breast cancer
#Both columns in Measures of Birth and Death

brst_cancer_over_40_df = measures_birth_Death_df[['CHSI_County_Name','CHSI_State_Name','Over_40','Brst_Cancer']].copy()
brst_cancer_over_40_df = brst_cancer_over_40_df.replace([i for i in NanList],np.NAN)

#Merge Demographic and brst_cancer_over_40_df
merged_demo_risk_factor_LeadCauseDeath = pd.merge(brst_cancer_over_40_df,Demo_df_populationSize,how='left',on=brst_cancer_over_40_df.index)
merged_demo_risk_factor_LeadCauseDeath.fillna(0,inplace=True)
merged_demo_risk_factor_LeadCauseDeath

over_40_population = [(x*y)/100 for x,y in zip(merged_demo_risk_factor_LeadCauseDeath['Over_40'],merged_demo_risk_factor_LeadCauseDeath['Population_Size'])]

merged_demo_risk_factor_LeadCauseDeath['Over_40_pop'] = pd.Series(over_40_population,merged_demo_risk_factor_LeadCauseDeath.index)
merged_demo_risk_factor_LeadCauseDeath

corrs = (merged_demo_risk_factor_LeadCauseDeath[['Over_40_pop', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_demo_risk_factor_LeadCauseDeath.Brst_Cancer)
        .rename(columns={'Over_40_pop' : 'Corr_Coef'}))
corrs_birthover40_BrstCancer = corrs[corrs['Corr_Coef'].notnull()]

#birth defects v/s infant mortality
infantMort_df = measures_birth_Death_df[['CHSI_County_Name','CHSI_State_Name','Infant_Mortality']].copy()
infantMort_df
infantMort_df = infantMort_df.replace([i for i in NanList],np.NAN)

merged_premature_LeadCauseBirthDef = pd.merge(df_leadDeathCauses_bf,infantMort_df['Infant_Mortality'],how='left',on=df_leadDeathCauses_bf.index)
del merged_premature_LeadCauseBirthDef['key_0']
merged_demo_premature_LeadCauseBf = pd.merge(merged_premature_LeadCauseBirthDef,Demo_df_populationSize,how='left',on=merged_premature_LeadCauseBirthDef.index)
merged_demo_premature_LeadCauseBf.fillna(0,inplace=True)
merged_demo_premature_LeadCauseBf

white_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Wh_BirthDef'],merged_demo_risk_factor_LeadCauseDeath['White Population'])]

black_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Bl_BirthDef'],merged_demo_premature_LeadCauseBf['Black Population'])]

hispanic_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Hi_BirthDef'],merged_demo_premature_LeadCauseBf['Hispanic Population'])]

other_population_bf = [((x)*y)/100 for x,y in zip(merged_demo_premature_LeadCauseBf['A_Ot_BirthDef'],merged_demo_premature_LeadCauseBf['Other Population'])]

merged_demo_premature_LeadCauseBf['White_Population_BirthDef'] = pd.Series(white_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Black_Population_BirthDef'] = pd.Series(black_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Hispanic_Population_BirthDef'] = pd.Series(hispanic_population_bf,index=merged_demo_premature_LeadCauseBf.index)
merged_demo_premature_LeadCauseBf['Other_Population_BirthDef'] = pd.Series(other_population_bf,index=merged_demo_premature_LeadCauseBf.index)


tot_population_Bf = [((a+b+c+d)/e)*100 for a,b,c,d,e in zip(merged_demo_premature_LeadCauseBf['White_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Black_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Hispanic_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Other_Population_BirthDef'],merged_demo_premature_LeadCauseBf['Population_Size'])]
merged_demo_premature_LeadCauseBf['Percentage_Total_Bf_Population'] = pd.Series(tot_population_Bf,index = merged_demo_premature_LeadCauseBf.index)

corrs = (merged_demo_premature_LeadCauseBf[['Infant_Mortality', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_demo_premature_LeadCauseBf.Percentage_Total_Bf_Population)
        .rename(columns={'Infant_Mortality' : 'Corr_Coef'}))
corrs_infant = corrs[corrs['Corr_Coef'].notnull()]

#Plotting graphs
plt.rcParams["figure.figsize"] = (15,15)
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax1 = sns.barplot(x=corrs_depression.index,y=corrs_depression['Corr_Coef'],palette='viridis')
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Co-relation between Major Depression and Suicide')
plt.title('Major Depression v/s Suicide', fontsize=20)

ax2 = fig.add_subplot(gs[0, 1])
ax2 = sns.barplot(x=corrs_Bf.index,y=corrs_Bf['Corr_Coef'],palette='YlOrBr')
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Co-relation between Birth Defects and Premature Deliveries')
plt.title('Birth Defects v/s Premature Deliveries',fontsize=20)



ax3 = fig.add_subplot(gs[1,0])
ax3 = sns.barplot(x=corrs_birthover40_BrstCancer.index,y=corrs_birthover40_BrstCancer['Corr_Coef'],palette='icefire')
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Co-relation between birth to women over 40 and Breast Cancer')
plt.title('Birth to women over 40 v/s Breast Cancer',fontsize=20)

ax4 = fig.add_subplot(gs[1, 1])
ax4 = sns.barplot(x=corrs_infant.index,y=corrs_infant['Corr_Coef'],palette='Wistia')
plt.xticks(rotation=90)
plt.xlabel('States')
plt.ylabel('Co-relation between infant mortality and birth defects')
plt.title('Infant mortality v/s Birth defects',fontsize=20)


#tight layout
plt.tight_layout()

