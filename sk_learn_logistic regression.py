import pandas as py 
from sklearn.linear_model import LogisticRegression

df=read_csv(data.csv) 

df['male']= df['sex']=='male' #convert all columns to numerical values 
X=df[['Pclass','male','age','siblings/spouses','parents/children','fare']].values 
#convert all interested predictive variables into X array 

y=df['survived'].values
#put all target in array 

model=LogisticRegression()

model.fit(X,y) 

print(model.coef_, model.intercept_)

model.predict(X[:5])
#return 1d array

