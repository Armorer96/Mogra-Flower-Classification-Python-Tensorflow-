
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
#hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

#sens=75                                  #white color
#lower_white=np.array([0,0,255-sens])
#upper_white=np.array([[200,sens-20,255]])
'''
lower_white=np.array([0,0,255-sens])
upper_white=np.array([[255,255,255]])
'''
'''
mask=cv2.inRange(hsv,lower_white,upper_white)
res=cv2.bitwise_and(image,image,mask=mask)
cv2.imshow('geay',gray)
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(res,kernel,iterations=1)
dilate=cv2.dilate(res,kernel,iterations=1)
#close and open for false positives and false negatives ...suitable for mogra
opening=cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)
#closing=cv2.morphologyEx(res,cv2.MORPH_CLOSE,kernel)
'''
edged = cv2.Canny(gray,50,100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
#cv2.imshow('edgy',edged)
cv2.waitKey(0) 
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]



(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over the contours individually
for c in cnts:
	
	if cv2.contourArea(c) < 100:
		continue

	
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	
	
	
	
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	
	
	
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	
	
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	
	
	
	
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	print("size in ppm",dimA, dimB )
	print("Eucledian Distance",dA, dB )

    	
	
	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	


	cv2.imshow("Image", orig)
	cv2.waitKey(0)
