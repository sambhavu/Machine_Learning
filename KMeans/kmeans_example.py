import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt


train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)


train.fillna(train.mean(), inplace = True)
test.fillna(test.mean(), inplace = True)

train = train.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1)
test = test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

labelencoder = LabelEncoder()
labelencoder.fit(train['Sex'])
labelencoder.fit(test['Sex'])

train['Sex'] = labelencoder.transform(train['Sex'])
test['Sex'] = labelencoder.transform(test['Sex'])

X = np.asarray(train.drop(['Survived'], 1).astype(float))
Y = np.asarray(train['Survived'])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(X_scaled)

correct = 0

for i in range(len(X)):
    predictme = np.array(X[i].astype(float))
    predictme = predictme.reshape(-1, len(predictme))
    prediction = kmeans.predict(predictme)

    if prediction == Y[i]:
        correct = correct +1

print(correct/len(X))








