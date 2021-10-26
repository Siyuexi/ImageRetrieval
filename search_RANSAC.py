#python search.py -i dataset/train/ukbench00000.jpg

import argparse as ap
import cv2
import numpy as np
import os
import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import numpy as np

from pylab import *
from PIL import Image
from rootsift import RootSIFT

# -----------RANSAC----------#
def RANSAC(rank_ID):
	isReRanking = True
	reRankingDepth = 30
	if isReRanking:
		idx = [[] for i in range(len(image_paths))]
		for i in range(reRankingDepth):
			for j in range(len(words)):
				index = rank_ID[i]
				for k in range(len(BoWs[index])):
					if words[j] == BoWs[index][k]:
						idx[i].append([j, k])
					else:
						continue

		reRankScores = []
		for i in range(reRankingDepth):
			query = []
			scene = []
			for j in range(len(idx[i])):
				# print(f"{idx[i][j][0]}_,_{idx[i][j][1]}")
				query.append(kpts[idx[i][j][0]].pt)
				scene.append(coordinates[rank_ID[i]][idx[i][j][1]])

			H, mask = cv2.findHomography(np.array(query), np.array(scene), cv2.RANSAC, 4)
			inliers_cnt, outliers_cnt = 0, 0
			for j in range(mask.shape[0]):
				if mask[j][0] == 1:
					inliers_cnt += 1
				else:
					outliers_cnt += 1
			index = rank_ID[i]
			score = 1.0 * inliers_cnt / len(idx[i])
			reRankScores.append([index, score])
		reRankScores = np.array(reRankScores)
		reRankScores = reRankScores[(reRankScores[:, 1]*(-1)).argsort()]
		rank_ID = reRankScores[:, 0].astype(int)
	return rank_ID

#----------Visualize the results----------#
def show_result(im, rank_ID):
	global numShow
	figure()
	gray()
	subplot(5,4,1)
	imshow(im[:,:,::-1])
	axis('off')
	for i, ID in enumerate(rank_ID[0:numShow]):
		img = Image.open(image_paths[ID])
		gray()
		subplot(5,4,i+5)
		imshow(img)
		axis('off')
	show()

if __name__ == '__main__':
	# Get the path of the training set
	parser = ap.ArgumentParser()
	parser.add_argument("-i", "--image", help="Path to query image", default="dataset/testing/all_souls_000000.jpg")
	args = vars(parser.parse_args())

	# Get query image path
	image_path = args["image"]

	# Load the classifier, class names, scaler, number of clusters and vocabulary
	im_features, image_paths, idf, numWords, voc = joblib.load("bag-of-words.pkl")
	coordinates, BoWs = joblib.load("information.pkl")
	inverted_bag = joblib.load("inverted_index.pkl")
	numShow = 16
	# Create feature extraction and keypoint detector objects
	# fea_det = cv2.FeatureDetector_create("SIFT")
	# des_ext = cv2.DescriptorExtractor_create("SIFT")
	fea_det = cv2.SIFT_create()

	# List where all the descriptors are stored
	des_list = []

	im = cv2.imread(image_path)

	im_size = im.shape
	# print str(im.shape)
	im = cv2.resize(im,(int(im_size[1]/4), int(im_size[0]/4)))


	# kpts = fea_det.detect(im)
	# kpts, des = des_ext.compute(im, kpts)
	kpts, des = fea_det.detectAndCompute(im, None)
	# rootsift
	#rs = RootSIFT()
	#des = rs.compute(kpts, des)

	des_list.append((image_path, des))

	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]

	#
	test_features = np.zeros((1, numWords), "float32")
	words, distance = vq(descriptors,voc)
	for w in words:
		test_features[0][w] += 1

	# print(words)

	# Perform Tf-Idf vectorization and L2 normalization
	test_features = test_features*idf
	test_features = preprocessing.normalize(test_features, norm='l2')




	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score)[0]
	show_result(im, rank_ID)
	rank_ID = RANSAC(rank_ID)
	show_result(im, rank_ID)
