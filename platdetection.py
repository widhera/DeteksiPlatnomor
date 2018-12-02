import cv2 as cv
import numpy as np
from skimage import feature
from skimage import exposure
import time
import imutils
from lbp import LBP
from imutils import paths
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from imutils import paths
import cPickle




def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image

	while True:

		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 

		yield image


def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


desc = LBP(24,8)
data = []
labels = []

img_rgb = cv.imread("IMG_20181201_190714.jpg")
img_rgb = cv.resize(img_rgb,(720,538))
h,w = img_rgb.shape[:2]
winW=w
winH=267

img = cv.imread("IMG_20181201_190714.jpg",0)
img = cv.resize(img_rgb,(720,538))
cv.imshow("Gray",img)
cv.waitKey(1)
time.sleep(0.025)

img_numpy = np.array(img)
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
median = cv.medianBlur(hogImage,5)
i=0


desc = LBP(24,8)
filename = 'model.joblib.pkl'
model = joblib.load(filename)


for resized in pyramid(img, scale=1.5):
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 		img_test = img_rgb[y:y + winH,x:x + winW]

 		gray = cv.cvtColor(img_test,cv.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		prediction = model.predict(hist.reshape(1,-1))

		clone = resized.copy()
		cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv.imshow("Window2", clone)

		if(prediction[0].split("\\")[1] == "iya"):
			cv.putText(img_rgb, prediction[0],(10,30), cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
			cv.imwrite('contoh/platnomor_'+str(i)+'.png',img_rgb[y:y + winH,x:x + winW])

		cv.imshow("Window", img_rgb[y:y + winH,x:x + winW])
		
		i +=1
		if(prediction[0].split("\\")[1] == "iya"):
			cv.waitKey(0)
		else:
			cv.waitKey(1)
			time.sleep(0.025)
