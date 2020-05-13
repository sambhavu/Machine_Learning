import numpy as np 

data = [20, 35, 64, 98, 103, 23, 45] 

print("mean: ", np.mean(data))
print("median: ", np.median(data))
print("50th percentile: ",np.percentile(data,50))
print("25th percentile: ",np.percentile(data,25))
print("75th percentile: ",np.percentile(data,75))
print("standard deviation: ",np.std(data))
print("variance: ",np.var(data)) 

