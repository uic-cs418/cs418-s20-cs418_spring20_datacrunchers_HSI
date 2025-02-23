# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:40:29 2020

@author: Varun
"""




merged_df_vul_suicide = pd.merge(df_Suicide_lead_cause,vulnerable_pop_and_env_health_df['Major_Depression'],how='left',on=df_Suicide_lead_cause.index)
del merged_df_vul_suicide['key_0']
merged_df_vul_suicide = pd.merge(merged_df_vul_suicide,Demo_df_populationSize,how='left',on=merged_df_vul_suicide.index)
merged_df_vul_suicide.fillna(0,inplace=True)

white_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Wh_Suicide'],merged_df_vul_suicide['D_Wh_Suicide'],merged_df_vul_suicide['White Population'])]

black_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Bl_Suicide'],merged_df_vul_suicide['D_Bl_Suicide'],merged_df_vul_suicide['Black Population'])]


hispanic_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Hi_Suicide'],merged_df_vul_suicide['D_Hi_Suicide'],merged_df_vul_suicide['Hispanic Population'])]

other_population_Suicide = [((a+b)*y)/100 for a,b,y in zip(merged_df_vul_suicide['C_Ot_Suicide'],merged_df_vul_suicide['D_Ot_Suicide'],merged_df_vul_suicide['Other Population'])]

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

#Birth Defects v/s Premature deliveries
measures_birth_Death_df = pd.read_csv('MEASURESOFBIRTHANDDEATH.csv')

premature_df = measures_birth_Death_df[['CHSI_County_Name','CHSI_State_Name','Premature']].copy()
premature_df = premature_df.replace([i for i in NanList],np.NAN)

df_leadDeathCauses_bf = df_leadDeathCauses[['State_FIPS_Code','County_FIPS_Code','CHSI_County_Name','CHSI_State_Name','A_Wh_BirthDef','A_Bl_BirthDef','A_Hi_BirthDef','A_Ot_BirthDef']].copy()
NanList = [-9999,-2222,-2222.2,-2,-1111,-1111.1,-9998.9]
df_leadDeathCauses_bf = df_leadDeathCauses_bf.replace([i for i in NanList],np.NAN)


merged_premature_LeadCauseBirthDef = pd.merge(df_leadDeathCauses_bf,premature_df['Premature'],how='left',on=df_leadDeathCauses_bf.index)
del merged_premature_LeadCauseBirthDef['key_0']
merged_demo_premature_LeadCauseBf = pd.merge(merged_premature_LeadCauseBirthDef,Demo_df_populationSize,how='left',on=merged_premature_LeadCauseBirthDef.index)
merged_demo_premature_LeadCauseBf.fillna(0,inplace=True)


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

over_40_population = [(x*y)/100 for x,y in zip(merged_demo_risk_factor_LeadCauseDeath['Over_40'],merged_demo_risk_factor_LeadCauseDeath['Population_Size'])]

merged_demo_risk_factor_LeadCauseDeath['Over_40_pop'] = pd.Series(over_40_population,merged_demo_risk_factor_LeadCauseDeath.index)

corrs = (merged_demo_risk_factor_LeadCauseDeath[['Over_40_pop', 'CHSI_State_Name_x']]
        .groupby('CHSI_State_Name_x')
        .corrwith(merged_demo_risk_factor_LeadCauseDeath.Brst_Cancer)
        .rename(columns={'Over_40_pop' : 'Corr_Coef'}))
corrs_birthover40_BrstCancer = corrs[corrs['Corr_Coef'].notnull()]

#birth defects v/s infant mortality
infantMort_df = measures_birth_Death_df[['CHSI_County_Name','CHSI_State_Name','Infant_Mortality']].copy()
infantMort_df = infantMort_df.replace([i for i in NanList],np.NAN)

merged_premature_LeadCauseBirthDef = pd.merge(df_leadDeathCauses_bf,infantMort_df['Infant_Mortality'],how='left',on=df_leadDeathCauses_bf.index)
del merged_premature_LeadCauseBirthDef['key_0']
merged_demo_premature_LeadCauseBf = pd.merge(merged_premature_LeadCauseBirthDef,Demo_df_populationSize,how='left',on=merged_premature_LeadCauseBirthDef.index)
merged_demo_premature_LeadCauseBf.fillna(0,inplace=True)

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