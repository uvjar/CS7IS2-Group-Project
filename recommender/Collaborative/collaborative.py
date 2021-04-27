import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import math 

reader = Reader(rating_scale=(0.5, 5.0), line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file("./datasets/ratings10M.csv", reader=reader)

#Ndata = []
#print(len(data.raw_ratings))
#print(type(data.raw_ratings))
#for i in range(100000):
#    Ndata.append(data.raw_ratings[i])
#print(Ndata[0])
#print(len(Ndata))
#print(type(Ndata))
#data.raw_ratings = Ndata

N = 1000000
avg = 0
for i in range(N):
    avg += data.raw_ratings[i][2]
avg = avg/N
print(avg)
sum = 0
for i in range(1000000,2000000):
    sum += (avg - data.raw_ratings[i][2])**2
sum = math.sqrt(sum/N)
print(sum)

sim_options = {
    "name": ["msd"],
    "min_support": [3],
    "user_based": [False],
}

param_grid = {"sim_options": sim_options}

gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse"], cv=5)
gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])


