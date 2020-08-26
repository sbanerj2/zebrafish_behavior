import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

## Code for running the detection and optical flow method together for detecting and 
## following the fish. Please change the paths to reflect the location of
## class name file (classesFile), model config (modelConfiguration) and
## saved weights for detection (modelWeights)
## location for downloading the these files:
## https://drive.google.com/drive/folders/1PzEGkGjYl60YpwnXQcOoTh7ruFsBCIWG
## To run: python code_modfied_opticalflow_final.py 
## Also creates two new directories (Input, Optical) under escape_data.
## The Optical folder is used in the subsequent program to create trajectory images


# Initialize the parameters
confThreshold = 0.96  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# Load names of classes
classesFile = "/path/to/model_weights/zebrafish.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
# yolov3-zebrafish-test-v3.cfg
modelConfiguration = "/path/to/model_weights/yolov3-zebrafish-test-v3.cfg"
modelWeights = "/path/to/model_weights/yolov3-zebrafish-train-v3_final.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def drawPred(mask_img, classId, conf, left, top, right, bottom):
    # Draw a bounding box. mask_img1 --> rectangle; mask_img -->circle
    #mask_img1 = mask_img
    #cv.rectangle(mask_img1, (left, top), (right, bottom), (255,255,255), -1)
    width = right - left
    height = bottom - top
    center_x = int(left + width / 2)
    center_y = int(top + height / 2)
    CIRCLE_RADIUS = 10
    cv.circle(mask_img, (center_x, center_y), CIRCLE_RADIUS, (255,255,255), -1 )
    return mask_img
    
    

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    ## New code
    mask_img = np.zeros((frameHeight,frameWidth), np.uint8)
    #mask_img2 = np.zeros((frameHeight,frameWidth), np.uint8)

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                print("Frame Width and height:", frameWidth, frameHeight)
                print(detection[0], detection[1], detection[2], detection[3])
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                print(center_x, center_y, width, height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        mask_img = drawPred(mask_img, classIds[i], confidences[i], left, top, left + width, top + height)
    return mask_img

## Input folder
video_path1 = '/path/to/escape_data/wetbench_videos' ##Can be downloaded from the link above

video_list = os.listdir(video_path1)
for each_video in video_list:
	video_name = each_video.split(".")[0]
	print ("Video_name: %s" %(video_name))
	frames_destination = "/path/to/escape_data/Videos/%s/Input"%(video_name) ##New locations created under escape_data
	optflow_destination = "/path/to/escape_data/Videos/%s/Optical"%(video_name) ##New locations created under escape_data
	#det_destination = "/afs/crc.nd.edu/user/s/sbanerj2/Paula_Videos/escape_data/Videos/%s/Detection"%(video_name)
	if not os.path.exists(frames_destination):
		os.makedirs(frames_destination)
	if not os.path.exists(optflow_destination):
		os.makedirs(optflow_destination)
	#if not os.path.exists(det_destination):
		#os.makedirs(det_destination)

	# The video feed is read in as a VideoCapture object
	cap = cv.VideoCapture(os.path.join(video_path1, each_video))
	# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
	ret, first_frame = cap.read()
	frame_name = video_name + "_0" + ".jpg"
	#det_frame_name = "det_" + video_name + "_0" + ".jpg"
	cv.imwrite(os.path.join(frames_destination, frame_name), first_frame)
	# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
	blob_prev = cv.dnn.blobFromImage(first_frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
	net.setInput(blob_prev)
	outs = net.forward(getOutputsNames(net))
	mask_img = postprocess(first_frame, outs)
	#first_frame1 = first_frame
	first_frame = cv.bitwise_and(first_frame, first_frame, mask=mask_img)
	#first_frame1 = cv.bitwise_and(first_frame1, first_frame1, mask=mask_img1)
	#cv.imwrite(os.path.join(det_destination, det_frame_name), first_frame1)
	prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
	# Creates an image filled with zero intensities with the same dimensions as the frame
	mask = np.zeros_like(first_frame)
	# Sets image saturation to maximum
	mask[..., 1] = 255
	count = 1
	while(ret):
		frame_name = video_name + "_" + str(count) + ".jpg"
		opt_frame = "opt_" + video_name + "_" + str(count) + ".jpg"
		#det_frame_name = "det_" + video_name + "_" + str(count) + ".jpg"
		#cv.imwrite(os.path.join(frames_destination, frame_name), prev_gray)
		ret, frame = cap.read()
		#cv.imwrite(os.path.join(frames_destination, frame_name), frame)
		if frame is None:
			break
		cv.imwrite(os.path.join(frames_destination, frame_name), frame)
		blob_present = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
		net.setInput(blob_present)
		outs_present = net.forward(getOutputsNames(net))
		mask_img_present  = postprocess(frame, outs_present)
		#frame1 = frame
		frame = cv.bitwise_and(frame, frame, mask=mask_img_present)
		#frame1 = cv.bitwise_and(frame1, frame1, mask=mask_img_present1)
		#cv.imwrite(os.path.join(det_destination, det_frame_name), frame1)
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
		mask[..., 0] = angle * 180 / np.pi / 2
		mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
		rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
		cv.imwrite(os.path.join(optflow_destination, opt_frame), rgb)
		prev_gray = gray
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
		count = count + 1
	print ("No. frames: %s"%(count))
	cap.release()
	cv.destroyAllWindows()
print ("Script ended")

