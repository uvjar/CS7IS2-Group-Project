from lenskit.algorithms.basic import Bias, Popular, TopN 
from lenskit import topn
from lenskit.metrics.predict import rmse
from lenskit.datasets import ML100K
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd
import os, sys

class MatrixFactorization:
	"""docstring for MatrixFactorization"""
	def __init__(self):
		pass
		
	def testPath(self):
		print(os.getcwd())

	def test(self,path):
		algo_pop = Bias(); 
		algo_als5 = als.BiasedMF(5)

		def eval(aname, algo, train, test, all_preds): 
		    fittable = util.clone(algo)
		    fittable = Recommender.adapt(fittable) 
		    fittable.fit(train)
		    # predict ratings
		    preds = batch.predict(fittable, test) 
		    preds['Algorithm'] = aname 
		    all_preds.append(preds)

		ml100k = ML100K(path)
		ratings = ml100k.ratings; 
		print(ratings.head())

		all_preds = []; test_data = []
		for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)): 
		    test_data.append(test)
		    eval('MF', algo_als5, train, test, all_preds)
		preds = pd.concat(all_preds, ignore_index=True)
		preds_mf= preds[preds['Algorithm'].str.match('MF')]
		test_data = pd.concat(test_data, ignore_index=True)
		print('RMSE MF:', rmse(preds_mf['prediction'],preds_mf['rating']))


 

