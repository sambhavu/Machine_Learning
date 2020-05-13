import pandas as pd 

df=pd.readcsv_csv(data.csv)
print(df.head) 
print(df.describe())

price = df['Price'].values #array of stock prices into array from df 
print(price.shape) # print size of array 
