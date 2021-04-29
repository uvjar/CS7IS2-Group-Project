
import argparse
import os
from MF import matrix_factorization
from ContentBased import content_based



def main():
	print("main")
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="test")
	parser.add_argument('-p','--path',metavar='DATASET_PATH',
		default='data',help="dataset path",type=str)
	parser.add_argument('-s','--size',metavar='DATASET_SIZE',
		default='100k',help="dataset size",type=str)
	parser.add_argument('-a','--algorithm',metavar='ALGORITHM',
		help="Select an algorithm to test. ",type=str,required=True)
	args = parser.parse_args()

	path = os.path.join(args.path,"ml-"+args.size)
	if not path:
		print("Dataset path does not exist")
		exit()

	if args.algorithm =='mf':
		al = matrix_factorization.MatrixFactorization()
		al.test(path)
		pass
	elif args.algorithm == 'item-based':
		pass
	elif args.algorithm == 'cb':
		al = content_based.ContentBased()
		al.test(path)
		pass
	else:
		print("Algorithm not exisit")
		exit()
	

	main()



