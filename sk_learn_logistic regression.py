import pandas as py 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df=read_csv(data.csv) 

df['male']= df['sex']=='male' #convert all columns to numerical values 
X=df[['Pclass','male','age','siblings/spouses','parents/children','fare']].values 
#convert all interested predictive variables into X array 

y=df['survived'].values
#put all target in array 

model=LogisticRegression()

model.fit(X,y) 

print(model.coef_, model.intercept_)

y_predict=model.predict(X)
#return 1d array

print("Accuracy: ", accuracy_score(y,y_predict))
print("Precision: ", presicion_score(y,y_predict)) 
print("Recall: ",recall_score(y,y_predict)) 
print("F1 score: ",f1_score(y,y_predict)) 


