import pandas as pd 
from sklearn.linearmodel import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score 

df=pd.read_csv('data.csv')
df['male']=df['Sex']=='male'
X=df[['Pclass', 'Male', 'Age', 'Sibling/spouses', 'Parents/Children', 'Fare']].values 
y=df['Survived'].values 

X_train,X_test,y_train,y_test = train_test_split(X,y) 

model1=LogisticRegression() 
model1=fit(X_train,y_train) 

y_pred_proba1=model1.predict_proba(X_test) 
print("model1 auc score: ",roc_auc_score(y_test,y_predict_proba1[:1])) 

model2=LogistricRegression() 
model2=fit(X_train[:,0:2],y_train) 
y_predict_proba2=model2.predict_proba(X_test[:,0:2]) 

print("Model2 auc score: ",roc_auc_score(y_test,y_predict_proba[:,1]))




