from lenskit.algorithms.basic import Bias, Popular, TopN
from lenskit import topn
from lenskit.metrics.predict import rmse
from lenskit.datasets import ML100K, MovieLens, ML1M, ML10M
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd
import os, sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBased:
    """Test calculates the similarity between movies"""
    def __init__(self):
        pass

    def testPath(self):
        print(os.getcwd())

    def test(self, path):
        ml20m = MovieLens('data/ml-20m')
        mlsmall = MovieLens('data/ml-latest-small')
        movies = mlsmall.movies # pd.read_csv('\data\ml-100k\movies.csv', sep=',', encoding='latin-1')
        genome_scores = ml20m.tag_genome #pd.read_csv('\data\ml-100k\genome-scores.csv', sep=',',encoding='latin-1')
        ratings = mlsmall.ratings #pd.read_csv('\data\ml-100k\ratings.csv', sep=',', encoding='latin-1')

        genome_scores_np = genome_scores.values
        gen = [[0] * 1128] * int(len(genome_scores) / 1128)
        gen_rel = [0] * 1128
        movs = [0] * int(len(genome_scores) / 1128)
        for m in range(0, int(len(genome_scores) / 1128)):
            for tag in range(0, 1128):
                gen_rel[tag] = genome_scores_np[1128 * m + tag][2]
            movs[m] = genome_scores_np[m * 1128][0]
            gen[m] = gen_rel[:]

        # compute the cosine similarity for all movies against all movies
        #cos_sim = cosine_similarity(gen)

        def get_movie_recommendation(id_movie_to_match):
            idx_matched_top5 = np.argsort(cos_sim[id_movie_to_match])[-6:-1]

            j = 0
            movie_recommended = [id_movie_to_match] * 5
            for i in idx_matched_top5:
                idx = movies['title'].get(movies['movieId'] == movs[i])
                movie_recommended[j] = idx.values[0]
                #print(i, cos_sim[i][0])
                j += 1
            print(movie_recommended)

        def set_user_profiles(ratings, gen):
            #user profiles
            users, rated = np.unique(ratings['userId'].values, return_counts=True)  # only ok movies are selected
            movies_to_gen_pred = 50
            users_rated = [[0] * 1128] * len(users)
            counter = 0  # incremented by +rated by user i
            for i in range(0, len(users)):  # for each user
                user_id = users[i]
                rated_movies = rated[i]
                user_tags = [[0] * 1128] * rated_movies
                for j in range(0, rated_movies):  # get averaged genome tags relevance
                    movie_id = ratings['movieId'][counter]
                    counter += 1
                    # get the row of that movie id, if the movie has available genome tags
                    if movie_id in movs:
                        m = movs.index(movie_id)
                        user_tags[j] = gen[m]
                # for each user create profile
            #    user_tags = np.array(user_tags)







