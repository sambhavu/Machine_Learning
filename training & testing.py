import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 

df = pd.read_csv('data.csv')

df['male']=df['Sex']=='male'

X=df[['Pclass','male','Age','Sibling/spouses', 'parents/children','Fare']].values
y=df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X,y) 



i'm
