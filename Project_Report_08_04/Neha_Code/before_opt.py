import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


Deaths_df=pd.read_csv(r"C:\Users\NehaS\Desktop\CS418\MEASURESOFBIRTHANDDEATH.csv")
Deaths_IL=Deaths_df[Deaths_df['CHSI_State_Name']=='Illinois']
Death_Infant=Deaths_IL[['CHSI_County_Name','LBW','VLBW','Premature','Under_18','Over_40','Unmarried','Late_Care','Infant_Mortality']]
Death_Infant=Death_Infant.replace(-1111.1,np.NaN)

Death_Infant=Death_Infant[Death_Infant['Infant_Mortality'].notnull()]

X=np.array(Death_Infant[['Infant_Mortality','LBW','VLBW']])
model = KMeans(n_clusters=3, random_state=1)
model.fit(X)
pred = model.predict(X)
levels=['2','0','1']
pred_val=[levels[x] for x in pred]
Death_Infant['Death_Level']=pred_val


X1=Death_Infant[['LBW','VLBW','Premature','Under_18','Over_40','Unmarried','Late_Care']].to_numpy()
y=Death_Infant['Death_Level'].to_numpy()
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.20,random_state=9)
clf = MultinomialNB()
clf.fit(X1_train,y_train)
inf_pred=clf.predict(X1_test)
print("The initial accuracy of the Multinomial Naive Bayes Classifier on the clustered model is :")
print(accuracy_score(y_test,inf_pred))

