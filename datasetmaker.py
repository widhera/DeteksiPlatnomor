import cv2 as cv
import numpy as np
from skimage import feature
from skimage import exposure
import argparse
import time
import cv2
import imutils


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
		# yield (w, y, image[y:y + windowSize[1], winW])
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


img_rgb = cv.imread("Bismillah.jpg")
img_rgb = cv.resize(img_rgb,(720,538))
h,w = img_rgb.shape[:2]
winW=w
winH=267
img = cv.imread("Bismillah.jpg",0)
img = cv.resize(img_rgb,(720,538))
cv.imshow("Gray",img)
cv2.waitKey(1)
time.sleep(0.025)

print type(img)
img_numpy = np.array(img)
print img_numpy
(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
median = cv.medianBlur(hogImage,5)
i=0

for resized in pyramid(img, scale=1.5):
	for (x, y, window) in sliding_window(resized, stepSize=8, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window2", clone)
		cv2.imshow("Window", img_rgb[y:y + winH,x:x + winW])
		cv2.imwrite('contoh/bukan_j'+str(i)+'.png',img_rgb[y:y + winH,x:x + winW])
		i +=1
		cv2.waitKey(1)
		time.sleep(0.025)
