##find trajectories (white connected blobs in dark ackground) by connected components and resize image 
## This program creates the train_classifier dataset (which is already supplied).
## This program edits/crops the trajectory images created in the previous step(code_trajectory_final.py) 
## to reflect the trajectories by removing black pixels and only keeping the colored pixels
## through connected components.
## To run: python code_connected_final.py

import os
import sys
import cv2
from scipy import ndimage

base_path ='/path/to/escape_data/'
prev_fold = 'Videos/Trajectory_complete'
dest_fold = 'classifier_data/train_classifier'
dest_folder = os.path.join(base_path, dest_fold)
prev_folder = os.path.join(base_path, prev_fold)
if not os.path.exists(dest_folder):
	os.makedirs(dest_folder)

img_list = os.listdir(prev_folder)
print (img_list)

for each_image in img_list:
	img_path = os.path.join(prev_folder, each_image)
	image = cv2.imread(img_path)
	frameHeight = image.shape[0]
	frameWidth = image.shape[1]
	frameDepth = image.shape[2]
	image_grey = image
	image_grey = cv2.cvtColor(image_grey, cv2.COLOR_BGR2GRAY)
	#bw_img = cv2.threshold(image_grey,127,255,cv2.THRESH_BINARY)
	mask = image_grey > image_grey.mean()
	label_im, nb_labels =ndimage.label(mask)
	obj = ndimage.find_objects(label_im)
	x1=[]
	y1=[]
	x2=[]
	y2=[]
	for each_obj in range(0,len(obj)):
		slice_x, slice_y= obj[each_obj]
		x1.append(slice_x.start)
		y1.append(slice_y.start)
		x2.append(slice_x.stop)
		y2.append(slice_y.stop)
	x_min = min(x1)
	y_min = min(y1)
	x_max = max(x2)
	y_max = max(y2)
	dest_path = os.path.join(dest_folder, each_image)
	cv2.imwrite(dest_path, image[x_min:x_max, y_min: y_max, :])
	print("data shape: ", image[x_min:x_max, y_min: y_max, :].shape)
		
	
	


