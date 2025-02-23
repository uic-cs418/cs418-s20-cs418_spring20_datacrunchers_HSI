# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:52:02 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:44:24 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:25:14 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:11:54 2020

@author: Varun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:07:22 2020

@author: Varun
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:05:14 2020

@author: Varun
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:41:59 2020

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


from sklearn.model_selection import train_test_split
X1=[
 'No_Exercise',
 'Few_Fruit_Veg',
 'Obesity',
 'High_Blood_Pres',
 'Smoker',
 'Uninsured',
 
 'Elderly_Medicare',
 'Disabled_Medicare',
 'Prim_Care_Phys_Rate',
 'Dentist_Rate',
 
 'FluB_Rpt',
 'HepA_Rpt',
 'HepB_Rpt',
 'Meas_Rpt',
 'Pert_Rpt',
 'CRS_Rpt',
 'Syphilis_Rpt',
 'FluB_Rpt%',
 'HepA_Rpt%',
 'HepB_Rpt%',
 'Meas_Rpt%',
 'Pert_Rpt%',
 'CRS_Rpt%',
 'Syphilis_Rpt%',
 
 
 'Pap_Smear',
 'Mammogram',
 'Proctoscopy',
 'Pneumo_Vax',
 'Flu_Vac',
 'Pap_Smear%',
 'Mammogram%',
 'Proctoscopy%',
 'Pneumo_Vax%',
 'Flu_Vac%',
 
 
 'Population_Size',
 'Population_Density',
 'Poverty',
 
 'Age_19_Under',
 'Age_19_64',
 'Age_65_84',
 'Age_85_and_Over',
 
 'White',
 'Black',
 'Native_American',
 'Asian',
 'Hispanic',
 
 'No_HS_Diploma',
 'No_HS_Diploma%',
 
 'Unemployed',
 'Unemployed%',
 
 'Sev_Work_Disabled',
 'Sev_Work_Disabled%',
 
 'Major_Depression',
 'Major_Depression%',
 
 'Recent_Drug_Use',
 'Recent_Drug_Use%',
 
 'Ecol_Rpt',
 'Salm_Rpt',
 'Shig_Rpt',
 
 'Toxic_Chem',
 
 #'All_Death',
 'Health_Status',
 'Unhealthy_Days',
 
 'LBW',
 'VLBW',
 'Premature',
 'Under_18',
 
 'Total_Births',
 'Total_Deaths',
 
 'Total_Births%',
 'Total_Deaths%',
 
 'Over_40',
 'Unmarried',
 'Late_Care',
 
 'Infant_Mortality',
 'IM_Neonatal',
 'IM_Postneonatal',

 'Homicide',
 'Homicide%',
 ]

mlmodel=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.copy()
mlmodel_bak=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.copy()
custom_df=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.copy()


def gridsearch(X,y): 
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
   
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]  
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    print(random_grid)
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3)
    
    rf_random.fit(X_train, y_train)
    predictions = rf_random.predict(X_train)
    error=abs(predictions - y_train)
    ##########################################################
    print('GriSearch: RandomForest: Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, predictions)))
    
    predictions = rf_random.predict(X_valid)
    errors = abs(predictions - y_valid)
    print('GridSearch : Random Forest: Test Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
    return()

def machinelearning(X,y):
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3)
    from sklearn.ensemble import RandomForestRegressor
    ########without grid search########
    rf = RandomForestRegressor(n_estimators = 1000,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               criterion='mse')
    forest=rf
    
    #########with grid search###############
    #rfcv=gridsearch()
    ##########################################
    
    print("------------------Data Specs-------------------------")
    print("Amount of Training Data", len(X_train))
    print("Amount of Training Labels Data", len(y_train))
  
    print("Amount of Testing Data", len(X_valid))
    print("Amount of Testing Labels Data", len(y_valid)) 
    print("-------------------------------------------------------")
    rf.fit(X_train, y_train)
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]    
    cols=X.columns
    Newdf=pd.DataFrame()
    Newdf['cols']=list(cols)
    Newdf['indices']=list(indices)
    Newdf['importances']=list(importances)
    Newdf=Newdf.sort_values(['importances'], ascending=False)
    
    # Print the feature ranking
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(25)
    plt.bar(Newdf.cols, Newdf.importances)
    plt.title("Feature importances")
    plt.xticks(rotation=90)
    result=(rf.score(X_train, y_train), rf.oob_score_,rf.score(X_valid, y_valid))
    
   
    ###########################################################
    predictions = rf.predict(X_train)
    error=abs(predictions - y_train)
    ##########################################################
    BaselinePred=[np.median(y_train) for i in range(0,len(y_train))]
    print('Baseline : Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, BaselinePred)))
    print('RandomForest: Train Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, predictions)))
    
    predictions = rf.predict(X_valid)
    errors = abs(predictions - y_valid)
    BaselinePred=[np.median(y_train) for i in range(0,len(y_valid))]
    print('\n\nBaseline : Test Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, BaselinePred)))
    print('Random Forest: Test Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))
    print("\n\nFeature ranking:")
    #############################################################3   

    return(result,Newdf.cols[:5],rf)
    
def modelrun(colname,mlmodel,X1):
    mlmodel=mlmodel.dropna(subset=[colname])
    X=mlmodel[X1]
    X=X.fillna(X.median())
    y=mlmodel[colname].values.ravel()
    result,top5,rf=machinelearning(X,np.array(y))
    print("Attribute Predicted    ",colname)
    print("Predictor Columns\n",top5)
    return(colname,top5,rf)
def grisearchdriver(colname,mlmodel,X1):
    mlmodel=mlmodel.dropna(subset=[colname])
    X=mlmodel[X1]
    X=X.fillna(X.median())
    y=mlmodel[colname].values.ravel()
    gridsearch(X,np.array(y))
ToBePredicted=['ALE','Diabetes','Lung_Cancer','Brst_Cancer','Col_Cancer','MVA','Stroke', 'Suicide','CHD']

#saving the results of all the models together.
#list1=[]
#for i in range(0,len(ToBePredicted)):
#        temp=ToBePredicted[i]
#        colname,top5,rf=modelrun(temp,custom_df,X1)
#        list1.append([rf,colname,list(top5)])
#ToBePredicted=['ALE','Diabetes','Lung_Cancer','Brst_Cancer','Col_Cancer','MVA','Stroke', 'Suicide','CHD']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

#df=pd.read_csv(r"PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.csv")
df=custom_df
LeadingCauseCols=['ALE','Diabetes','Lung_Cancer','Lung_Cancer%','Brst_Cancer%','Brst_Cancer','Col_Cancer','Col_Cancer%','MVA','Stroke', 'Suicide','CHD']
data=df[LeadingCauseCols]
data = data.fillna(data.mean())
data = data.astype(float)
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
prd=cluster.fit_predict(data_scaled)
prd=list(prd)
result=df[['CHSI_County_Name_x','CHSI_State_Name_x']+LeadingCauseCols]
result['Cluster']=prd
tempdf=result.loc[result['Cluster']==3]
