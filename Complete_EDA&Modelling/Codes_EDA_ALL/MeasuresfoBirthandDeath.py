# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:45:54 2020

@author: Varun
"""

mypath=r'D:\Drive\Coursework\IDS\Project_IDS\Dataset'
os.chdir(mypath)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
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
#####################################################################################
PSU_Demo_VPEH_SMOH_RFAC_df=PSU_Demo_VPEH_SMOH_RFAC_df.drop(columns=['_merge'])
PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df=PSU_Demo_VPEH_SMOH_RFAC_df.merge(MOBAD, on=['State_FIPS_Code', 'County_FIPS_Code'], how='left', indicator=True)
####converttopercent###########
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
##################################################################3
CorrTable=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df.corr()
#IM
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(15)

ax1=fig.add_subplot(1,3,1)
MOBAD['Infant_Mortality'].hist(bins=100,ax=ax1)
ax1.set_title("Distribution of Homicide Variable")

ax2=fig.add_subplot(1,3,2)
MOBAD['IM_Neonatal'].hist(bins=100,ax=ax2)
ax2.set_title("Distribution of Infant_Mortality - Neonatal Variable")

ax3=fig.add_subplot(1,3,3)
MOBAD['IM_Postneonatal'].hist(bins=100,ax=ax3)
ax3.set_title("Distribution of Infant_Mortality - PostNeonantal Variable")

#Diseases
fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(15)

ax1=fig.add_subplot(2,3,1)
MOBAD['Brst_Cancer'].hist(bins=100,ax=ax1)
ax1.set_title("Distribution of Brst_Cancer Variable")

ax2=fig.add_subplot(2,3,2)
MOBAD['Col_Cancer'].hist(bins=100,ax=ax2)
ax2.set_title("Distribution of Col_Cancer Variable")

ax3=fig.add_subplot(2,3,3)
MOBAD['Lung_Cancer'].hist(bins=100,ax=ax3)
ax3.set_title("Distribution of Lung_Cancer Variable")

ax4=fig.add_subplot(2,3,4)
MOBAD['CHD'].hist(bins=100,ax=ax4)
ax4.set_title("Distribution of Coronary Heart Disease  Variable")

ax5=fig.add_subplot(2,3,5)
MOBAD['Suicide'].hist(bins=100,ax=ax5)
ax5.set_title("Distribution of Infant_Mortality - PostNeonantal Variable")

ax6=fig.add_subplot(2,3,6)
MOBAD['Stroke'].hist(bins=100,ax=ax6)
ax6.set_title("Distribution of Stroke Variable")

######plots

ax=sns.scatterplot('Total_Deaths', 'Uninsured', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Total Number of Deaths & Number of People UnInsured are Correlated (Log Scale)")

ax=sns.scatterplot('Smoker', 'Lung_Cancer', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax.set_title("Total Number of Lung Cancer Cases & Number of Smokers are Correlated ")

ax1=sns.scatterplot('Poverty', 'Homicide', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax1.set_title("Counties with more poverty percentage see more homicides\n Positively Correlated")


ax1=sns.scatterplot('Poverty', 'Under_18', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax1.set_title("Counties with more poverty percentage see more females under 18 giving birth \n Positively Correlated")
plt.ylabel('Females under 18 giving birth', fontsize=12)

ax1=sns.scatterplot('Poverty', 'Unmarried', data=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df)
ax1.set_title("Counties with more poverty percentage see more unmarried female giving birth \n Positively Correlated")
plt.ylabel('Unmarried Females  giving birth', fontsize=12)

Someplots=PSU_Demo_VPEH_SMOH_RFAC_MOBAD_df[['No_HS_Diploma','Unemployed','Major_Depression','Recent_Drug_Use','Total_Deaths']]
Someplots=pd.DataFrame(Someplots.corr())
Someplots=Someplots[['Total_Deaths']]
fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(5)
ax = sns.heatmap(Someplots,vmin=0, vmax=1,center=0 )
