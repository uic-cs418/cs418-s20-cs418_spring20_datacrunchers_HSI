from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Deaths_df=pd.read_csv(r"C:\Users\NehaS\Desktop\CS418\MEASURESOFBIRTHANDDEATH.csv")
riskdf=pd.read_csv(r"C:\Users\NehaS\Desktop\CS418\RISKFACTORSANDACCESSTOCARE.csv")
Demodf=pd.read_csv(r"C:/Users/NehaS/Desktop/CS418/DEMOGRAPHICS.csv")
Demodf=Demodf[Demodf['CHSI_State_Name']=='Illinois']
riskdf=riskdf.replace(-1111.1,np.NaN)
Illinoisdf=riskdf[riskdf['CHSI_State_Name']=='Illinois']
phys_level=pd.DataFrame(Illinoisdf[['CHSI_County_Name','Prim_Care_Phys_Rate']])
Deaths_IL=Deaths_df[Deaths_df['CHSI_State_Name']=='Illinois']
deaths_perpop=Deaths_IL.merge(Demodf['Population_Size'], how='left', on=phys_level['CHSI_County_Name'])
death_ratio=np.array([x/y*100 for x,y in zip(deaths_perpop['Total_Deaths'],deaths_perpop['Population_Size'])])
phys_level['Death_Ratio']=death_ratio

X=X=phys_level[['Prim_Care_Phys_Rate','Death_Ratio']].to_numpy()
model = KMeans(n_clusters=3, random_state=1)
model.fit(X)
pred = model.predict(X)

plt.scatter(X[:,0], X[:,1], c=model.labels_,cmap='rainbow_r', s=50)
plt.scatter(model.cluster_centers_[:,0] ,model.cluster_centers_[:,1], color='gold', marker='*', s=250, label='Cluster center')
plt.title("Accessibility to Healthcare to Death Ratio Plot", fontsize=20)
plt.xlabel('Physicians per 100k population')
plt.ylabel('Death Ratio')
plt.legend()

