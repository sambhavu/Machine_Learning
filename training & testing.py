import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from skleaarn.metrics import accuracy_score, recall_score, precision_score, f1_score


df = pd.read_csv('data.csv')

df['male']=df['Sex']=='male'

X=df[['Pclass','male','Age','Sibling/spouses', 'parents/children','Fare']].values
y=df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y) 


model=LogisticRegression() 
model=fit(X_train, y_train) 
y_predict=model.predict(X_test) 

print("precision: ", precision_score(y_test, y_predict)) 
print("accuracy: ", accuracy_score(y_test, y_predict)) 
print("recall: ", recall_score(y_test, y_predict))
print("f1: ", f1_score(y_test, y_predict))



