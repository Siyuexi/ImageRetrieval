import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import normalize
from functions.Opts import myopts
import functions.Utils as U
import os


#####################
BASE_DATABASE_PATH = 'dataset/training'
BASE_QUERY_PATH = 'dataset/testing'

U.branches = 5
U.maxDepth = 5
U.InitModel()
#####################

DescList = []
DescImgMap = []
ImgList = []
LeafImgList = []
# sift = cv2.xfeatures2d.SIFT_create(1000)
sift = cv2.SIFT_create(1000)



print("Extracting SIFT Features...")

training_names = os.listdir(BASE_DATABASE_PATH)

image_paths = []
for training_name in training_names:
    image_path = os.path.join(BASE_DATABASE_PATH, training_name)
    image_paths += [image_path]

DATA_SIZE = len(image_paths)

for i in range(DATA_SIZE):
	Path = image_paths[i]
	img = cv2.imread(Path)
	im_size = img.shape
	img = cv2.resize(img,(int(im_size[1]/4),int(im_size[0]/4)))
	if img is None:
		continue
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,des = sift.detectAndCompute(gray,None)
	DescList.append(des)
	ImgList.append(Path)
	LeafImgList.append({})
	print("{}/{} completed...".format(i+1,DATA_SIZE))

		
print("SIFT Feature Extraction Completed...")

print("Building The Tree...")

Root = U.BuildTree(-1, np.vstack(DescList), 0)
U.Root = Root

for i,desc in enumerate(DescList):
	for des in desc:
		Root.tfidf(des,image_paths[i])

Leaves = Root.allLeaves()

print("Built The Tree...")


print("Querying...")

Path = 'dataset/testing/bodleian_000192.jpg'
img = cv2.imread(Path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, des = sift.detectAndCompute(gray, None)
q = {leaf: 0 for leaf in Leaves}

for d in des:
	leaf = Root.query(d)
	q[leaf] = q[leaf] + 1 if leaf in q else 1

q = {y: y.weight() * z / sum(q.values()) for y, z in q.items()}

TOP_FIVE = [(-1e8, None) for i in range(5)]
for j, lidict in enumerate(LeafImgList):
	score = 0
	for leaf in Leaves:
		if leaf in lidict and leaf in q:
			score += q[leaf] * lidict[leaf]  # abs(q[leaf] - lidict[leaf])

	TOP_FIVE.append((score, image_paths[j]))
	TOP_FIVE = sorted(TOP_FIVE, key=lambda x: x[0])[::-1][:5]

TOP_FIVE = [x[1] for x in TOP_FIVE]
print(TOP_FIVE,file=open('tree_out.txt','w'))