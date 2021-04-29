import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
# from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from lenskit.algorithms.basic import Bias, Popular, TopN, Recommender
from lenskit import topn
from lenskit.metrics.predict import rmse
import lenskit.crossfold as xf



# Read in files
df_movies = pd.read_csv(r'C:\Users\darao\PycharmProjects\movie_recommender\movies.csv',
                        usecols=['movieId', 'title'],
                        dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    r'C:\Users\darao\PycharmProjects\movie_recommender\ratings.csv',
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Create easy to read pivot table where columns are userid and rows are movie id
df_pivot = df_ratings.pivot(index='movieId',columns='userId',values='rating')
df_pivot.fillna(0, inplace=True)
print(df_pivot.head())

# Remove noise from data
# Visualise votes per movie
no_user_voted = df_ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = df_ratings.groupby('userId')['rating'].agg('count')
f, ax = plt.subplots(1, 1, figsize=(16, 4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index, no_user_voted, color='b')
plt.axhline(y=10, color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.title("No. of user ratings per Movie")
plt.show()

# Remove movies with less than 10 votes
df_pivot = df_pivot.loc[no_user_voted[no_user_voted > 10].index, :]

# Visulaise votes per each user
f, ax = plt.subplots(1, 1, figsize=(16, 4))
plt.scatter(no_movies_voted.index,no_movies_voted, color='b')
plt.axhline(y=50, color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.title("No. of votes per User")
plt.show()

# Remove movies with less than 50 votes
df_pivot = df_pivot.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Remove sparsity
csr_data = csr_matrix(df_pivot.values)
df_pivot.reset_index(inplace=True)
print(df_pivot.head())

# Use KNN with cosine to compute similarity
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)


def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = df_movies[df_movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = df_pivot[df_pivot['movieId'] == movie_idx].index[0]

        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]

        recommend_frame = []

        for val in rec_movie_indices:
            movie_idx = df_pivot.iloc[val[0]]['movieId']
            idx = df_movies[df_movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': df_movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
        return df

    else:

        return "No movies found. Please check your input"

print(get_movie_recommendation('Iron Man'))


algo_pop = Bias()
algo_ii = knn.ItemItem(20)

def eval(aname, algo, train, test, all_preds):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    preds = batch.predict(fittable, test)
    preds['Algorithm'] = aname
    all_preds.append(preds)

all_preds = []
test_data = []
for train, test in xf.partition_users(df_ratings, 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    eval('BIAS', algo_pop, train, test, all_preds)
    eval('II', algo_ii, train, test, all_preds)

preds = pd.concat(all_preds, ignore_index=True)

preds_ii = preds[preds['Algorithm'].str.match('BIAS')]
print(preds_ii.head())

preds_bias = preds[preds['Algorithm'].str.match('BIAS')]
print(preds_bias.head())

test_data = pd.concat(test_data, ignore_index=True)

print('RMSE BIAS: ', rmse(preds_bias['prediction'], preds_bias['rating']))
print('RMSE II: ', rmse(preds_ii['prediction'], preds_ii['rating']))