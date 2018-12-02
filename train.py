from lbp import LBP
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from imutils import paths
import argparse
import cv2
import cPickle

ap = argparse.ArgumentParser()
ap.add_argument("-t","--training",required=True, help="path to the training images")
args = vars(ap.parse_args())

desc = LBP(24,8)
data = []
labels = []

#extract LBP

for imagePath in paths.list_images(args["training"]):
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	labels.append(imagePath.split("_")[0])
	data.append(hist)
model = LinearSVC(C=100.00,random_state=42)
model.fit(data,labels)
# print model
filename = 'model.joblib.pkl'  
_ = joblib.dump(model, filename, compress=9) 

