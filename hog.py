from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import numpy

print("[INFO] Extracting features...")
model = KNeighborsClassifier(n_neighbors = 1)

data = []
labels = []

for imagePath in paths.list_images('./hog/car_logos'):
	print("[INFO] Training classifier...")
	make = imagePath.split("\\")[-2]
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edged = imutils.auto_canny(gray)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	(x, y, w, h) = cv2.boundingRect(c)
	logo_pre = gray[y:y + h, x:x + w]
	logo = cv2.resize(logo_pre, (200, 100))
	H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
	data.append(H)
	labels.append(make)
	model.fit(data, labels)
	
for(i, imagePath) in enumerate(paths.list_images("./hog/test_images")):
	print("[INFO] Evaluating...")
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	logo = cv2.resize(gray, (200, 100))
	(H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", 
		visualize=True)
	pred = model.predict(H.reshape(1, -1))[0]
	hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
	hogImage = hogImage.astype("uint8")
	cv2.imshow("HOG Image #{}".format(i + 1), hogImage)
	cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
		(0, 255, 0), 3)
	cv2.imshow("Test Image #{}".format(i + 1), image)

while True:
	if cv2.waitKey(10) == 27:
		break