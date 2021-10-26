#!/usr/local/bin/python2.7
#python findFeatures.py -t dataset/training/

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
from rootsift import RootSIFT
import math

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
#train_path = "dataset/train/"

training_names = os.listdir(train_path)

numWords = 1000

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

# Create feature extraction and keypoint detector objects
# fea_det = cv2.FeatureDetector_create("SIFT")
# des_ext = cv2.DescriptorExtractor_create("SIFT")
fea_det = cv2.SIFT_create()

# List where all the descriptors are stored
des_list = []

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    im_size = im.shape

    # print str(im.shape)
    # if im_size[1] > im_size[0]:
    #     im = cv2.resize(im,(imagesize_0,imagesize_1))
    # else:
    #     im = cv2.resize(im,(imagesize_1,imagesize_0))
    # print str(im.shape)

    im = cv2.resize(im,(int(im_size[1]/4),int(im_size[0]/4)))

    print ("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))
    # kpts = fea_det.detect(im)
    # kpts, des = des_ext.compute(im, kpts)
    kpts, des = fea_det.detectAndCompute(im, None)
    # rootsift
    #rs = RootSIFT()
    #des = rs.compute(kpts, des)
    des_list.append((image_path, des))
    # print str(des.shape)  
    
# Stack all the descriptors vertically in a numpy array
downsampling = 4
descriptors = des_list[0][1][::downsampling,:]
for image_path, descriptor in des_list[1:]:
    # print np.size(descriptor)
    # print descriptor
    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

# Stack all the descriptors vertically in a numpy array
# descriptors = des_list[0][1]
# for image_path, descriptor in des_list[1:]:
#     print np.size(descriptor)
#     print descriptor
#     # if np.size(descriptor) != 0:
#     descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
print("Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1) 

# Calculate the histogram of features
# im_features = np.zeros((len(image_paths), numWords), "float32")
# for i in range(len(image_paths)):
#     words, distance = vq(des_list[i][1],voc)
#     for w in words:
#         im_features[i][w] += 1
words_index = [[] for i in range(numWords)]
nbr_occurences = np.zeros(numWords)
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    words_taboo = [0 for i in range(numWords)]
    for w in words:
        if(words_taboo[w]==0):
            words_index[w].append(i)
            words_taboo[w] += 1
            nbr_occurences[w] += 1
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# # Perform Tf-Idf vectorization
# nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# # Perform L2 normalization
# im_features = im_features*idf
# im_features = preprocessing.normalize(im_features, norm='l2')

# joblib.dump((im_features, image_paths, idf, numWords, voc), "bag-of-words.pkl", compress=3)   
joblib.dump((words_index, image_paths, idf, numWords, voc), "bag-of-words-inverted.pkl", compress=3)          