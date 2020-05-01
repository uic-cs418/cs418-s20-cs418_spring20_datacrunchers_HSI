from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import statsmodels.api as sm

Deaths_df=pd.read_csv(r"C:\Users\NehaS\Desktop\CS418\MEASURESOFBIRTHANDDEATH.csv")
Death_Infant=Deaths_df[['CHSI_County_Name','LBW','VLBW','Premature','Under_18','Over_40','Unmarried','Late_Care','Infant_Mortality']]
Death_Infant=Death_Infant.replace([-2222.2,-1111.1],np.NaN)
Death_Infant=Death_Infant[Death_Infant['Infant_Mortality'].notnull()]
Death_Infant=Death_Infant.dropna()
X=np.array(Death_Infant[['Infant_Mortality','LBW','VLBW','Premature','Over_40','Unmarried','Late_Care']])
model = KMeans(n_clusters=2, random_state=1)
model.fit(X)
pred = model.predict(X)
levels=['2','1']
pred_val=[levels[x] for x in pred]
Death_Infant['Death_Level']=pred_val



X1=Death_Infant[['LBW','VLBW','Premature','Under_18','Over_40','Unmarried','Late_Care']].to_numpy()
y=Death_Infant['Death_Level'].to_numpy()
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.10,random_state=5)
clf = MultinomialNB()
clf.fit(X1_train,y_train)
inf_pred=clf.predict(X1_test)
X_opt =X1[:,[0,1,2,3,4,5,6]]
OLS_res = sm.OLS(endog=Death_Infant['Infant_Mortality'], exog=X_opt).fit()
#print(OLS_res.summary())
print("\nAccuracy of the model on world data :")
print(accuracy_score(y_test,inf_pred))
