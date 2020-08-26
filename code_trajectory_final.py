import os
import sys
import PIL
import random
import numpy as np
from PIL import Image

### Create trajectory as an average image of the optical flow image
### Please change the paths: video_path1, inp_directory, out_directory
### The trajectory images are stored within escape_data/Videos/Trajectory_complete 

### To run: python code_trajectory_final.py

def create_average(video_name, inp_directory, out_directory):
	N = 119
	num = 1
	base_name = "opt_" + video_name + "_" + str(num) + ".jpg"
	base_path = os.path.join(inp_directory, base_name)
	w,h = Image.open(base_path).size
	
	arr = np.zeros((h,w,3),np.float32)
	
	for i in range(num, num+N):
		frame_name = "opt_" + video_name + "_" + str(i) + ".jpg"
		frame_path = os.path.join(inp_directory, frame_name)
		imarr = Image.open(frame_path)
		imarr = np.asarray(imarr)
		imarr = imarr.astype('float32')/255
		imarr= np.clip(imarr, 0., 1.)
		arr = arr+imarr
	
	arr = np.clip(arr, 0., 1.)
	arr = arr * 255
	arr = arr.astype('uint8')
	out = Image.fromarray(arr,mode="RGB")
	out_frame = "traj_" + video_name + ".jpg"
	out_path = os.path.join(out_directory, out_frame)
	out.save(out_path)

video_path1 = '/path/to/escape_data/wetbench_videos'
video_list = os.listdir(video_path1)
for each_video in video_list:
	video_name = each_video.split(".")[0]
	inp_directory = '/path/to/escape_data/Videos/%s/Optical'%(video_name)
	out_directory = '/path/to/escape_data/Videos/Trajectory_complete' ## New directory
	if not os.path.exists(out_directory):
		os.makedirs(out_directory)
	create_average(video_name, inp_directory, out_directory)

print ("Script ended, ")
#print(rand_int)
		
		
		
	


	
