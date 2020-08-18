import numpy as np;
import cv2;

def sign(value):
	return round(value/abs(value)) if value != 0 else 0;

def mean(values):
	return round(sum(values) / len(values), 2) if type(values) == list and len(values) > 0 else 0.0;

def normalizeImage(image):
	img = image.astype(np.float16);
	img /= 255.0;
	return img;

def centerImage(image):
	img = image.astype(np.float16);
	means = img.mean(axis=(0, 1), dtype=np.float16);
	img -= means;
	return img;

def normalizeCenterImage(image):
	return centerImage(normalizeImage(image));
	
def preprocessImage(image, resizeDim=None):
	if resizeDim != None:
		# image = cv2.blur(image, (3,3));
		image = cv2.resize(image, resizeDim, interpolation=cv2.INTER_AREA);
	
	image = cv2.blur(image, (3,3));
	image = normalizeImage(image);
	
	return image;