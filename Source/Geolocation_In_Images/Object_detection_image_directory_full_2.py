'''
Name: Object_detection_image_directory_full.py
By: Roberto Rodriguez
Date created: 2/17/2019
Last updated: 2/22/2019
Description:
This program uses a TensorFlow-trained classifier to perform object detection.
It loads the classifier and image data for a directory of images. It first breaks an image into segments and save them in a directry.
It draws boxes and scores around the objects of interest in the image segments.
It performs calculations on the centerpoints of the detection boxes and saves the geocoordinates to a CSV file.

Notes:
2/22/2019 - Fixed debug header
2/21/2019 - Moved geolocation calculations into loop (fixed exif reading error)
2/20/2019 - Added debug output and fixed intermediate image naming
'''

# Import packages
import os, sys
import cv2
import numpy as np
import tensorflow as tf
import glob
import math
import io
import pandas as pd
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
from numpy import genfromtxt
from PIL import Image, ExifTags

# Import custom libraries
from convert_to_degrees_2 import dms_to_degrees
from xmp_read import get_xmp
from WGS84toUTM import WGS84toUTM
from sensor_dim import get_sensor

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)

#Adding the directory
directory_path = askdirectory(initialdir="C:/",title='Choose a directory.')
#directory_path = 'C:/Users/Frost-Desktop/Desktop/Coordinate_Workflow/Test_Dir'

#Ask for information and initialize variables
Alt_Agl = simpledialog.askfloat("Input", "Enter altitude above ground level.", minvalue=0.0, maxvalue=1000.0)
#Alt_Agl = 30.0

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = askopenfilename(initialdir="C:/",title='Choose an inference graph.')
#PATH_TO_CKPT = 'C:/tensorflow2/Coconut_RP_2/object_detection/inference_graph_5000/frozen_inference_graph.pb'

# Path to label map file
PATH_TO_LABELS = askopenfilename(initialdir="C:/",title='Choose a label map.')
#PATH_TO_LABELS = 'C:/tensorflow2/Coconut_RP_2/Data/label_map.pbtxt'

# Number of classes the object detector can identify
NUM_CLASSES = simpledialog.askinteger("Input", 'Enter number of classes.', minvalue=1, maxvalue=1000)
#NUM_CLASSES = 1

# Score threshold
THRESHOLD = simpledialog.askfloat("Input", "Enter threshold score (0.0-1.0)", minvalue=0.0, maxvalue=1.0)
#THRESHOLD = 0.5

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
print('Loading Tensorflow model.')
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

	sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
print('Tensorflow model loaded.')

# Read list of images in directory
dir_list = []
for img_name in glob.glob((directory_path)+'/'+'*.jpg'): #Assuming JPG
	dir_list.append(img_name)

# Create an array to store coordinates and name of image they come from
pixel_coordinates = []
geocoordinates = []
image_name = []

# Debug array
# debug = []
	
for n in range(len(dir_list)):
	# Retreive image information
	print('Reading image data.')
	image_path = dir_list[n]
	PILFile=Image.open(image_path)

	#Get EXIF metadata from image
	exifData = {}
	exifDataRaw = PILFile._getexif()
	for tag, value in exifDataRaw.items():
		decodedTag = ExifTags.TAGS.get(tag, tag)
		exifData[decodedTag] = value

	#Get XMP string from image
	xmp = get_xmp(image_path)

	#Get yaw and pitch from XMP string
	c='"'
	enum=[pos for pos, char in enumerate(xmp) if char == c]

	yaw_str=xmp[enum[36]+1:enum[37]]
	#print(yaw_str)
	yaw=float(yaw_str)
	#print(yaw)

	pitch_str=xmp[enum[38]+1:enum[39]]
	#print(pitch_str)
	pitch=float(pitch_str)
	#print(pitch)

	#Convert to radians and define pitch as pointing down
	pitch = ((pitch + 90.0) * math.pi / 180.0)
	yaw = (yaw * math.pi / 180.0)
	if yaw < 0:
		yaw = yaw + 2 * math.pi
		#print(pitch, yaw)

	#Get camera and image properties
	ImageW = exifData['ExifImageWidth']
	ImageH = exifData['ExifImageHeight']
	focal = exifData['FocalLength']
	model = exifData['Model']
	(SensorW, SensorH) = get_sensor(model)

	#Get GPS Inforation
	gpsinfo = exifData['GPSInfo']
	lat = dms_to_degrees(gpsinfo[2])
	if gpsinfo[1]!= 'N':
		lat = 0 - lat
	lon = dms_to_degrees(gpsinfo[4])
	if gpsinfo[3] != 'E':
		lon = 0 - lon
	#print(lat,",",lon)

	#Convert Latitude and Longitude to UTM coordinates
	UTM = WGS84toUTM(lat,lon)

	# Checkpoint
	print('Loaded image data.')
	print('Segmenting image.')

	# Break image into segments and save in directory
	image_dir = image_path[0:-4]
	os.mkdir(image_dir)
	a=1 # Start counting at 1 to simplify later calculations
	b=1
	c=0
	d=0
	while d < 1000 * (ImageH//1000 + 1): 
		crop(image_path, (c, d, c+1000, d+1000), image_dir+'/'+str(b)+str(a)+'.jpg')
		if(c < 1000 * (ImageW//1000 + 1)):
			c=c+1000
			a=a+1
		if(c >= 1000 * (ImageW//1000 + 1)):
			c=0
			d=d+1000
			b=b+1
			a=1

	#Checkpoint
	print('Segmentation completed.')

	# Create an array containing all of the valid filenames
	image_list = []

	# Make a list of all valid images of the jpg format
	for filename in glob.glob((image_dir)+'/'+'*.jpg'): #Assuming JPG
		image_list.append(filename)
	#print(image_list)

	# Set segmented image dimensions
	im_height = 1000
	im_width = 1000

	# Checkpoint
	print('Scanning segments.')
	
	#Loop through all images in list of images
	for i in range(len(image_list)):
		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value
		image = cv2.imread(image_list[i])
		image_expanded = np.expand_dims(image, axis=0)
    
		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: image_expanded})

		# Draw the results of the detection (aka 'visulaize the results')
		vis_util.visualize_boxes_and_labels_on_image_array(
			image,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			category_index,
			use_normalized_coordinates=True,
			line_thickness=5,
			min_score_thresh=THRESHOLD)
		
		# Save proceesed image with results
		temp_image_name = image_list[i]
		cv2.imwrite(temp_image_name[:-4]+'R.jpg',image)
	
		# Progress update
		print('Completed '+str(i+1)+' of '+str(len(image_list))+' segments.')
    
		# Save boxes above the minimum threshold and normalize them
		for j, box in enumerate(np.squeeze(boxes)):
			if(np.squeeze(scores)[j] > THRESHOLD):
				y_coord = (box[0]*im_height + box[2]*im_height) / 2
				x_coord = (box[1]*im_width + box[3]*im_width) / 2
				x_coord = int(round(x_coord))
				y_coord = int(round(y_coord))
				local_frame = int(image_list[i][-6:-4])
				global_x = 1000 * (local_frame%10 - 1) + x_coord
				global_y = 1000 * (local_frame//10 - 1) + y_coord
				#coord = [local_frame, x_coord, y_coord, global_x, global_y]
				coord = [global_x, global_y]
				pixel_coordinates.append(coord)
				image_name.append(image_dir[-8:])
				u = global_x
				v = global_y
				#for flat model
                #calculate distance offset from principal point
                tan_delta_Y = ((SensorH/(ImageH-1))*((ImageH-1)/2-v))/focal
                delta_Y = math.atan(tan_delta_Y)
                Z_p = Alt_Agl * math.cos(delta_Y) / (math.cos(gamma + delta_Y))
                Y_p = -1* Alt_Agl * math.sin(delta_Y) / (math.cos(gamma + delta_Y))
                X_p = (1/focal)*(SensorW/(ImageW-1))*math.sqrt(Z_p*Z_p+Y_p*Y_p)*(u-(ImageW-1)/2)
                #perform rotation otherwise
                X = (-1 * X_p * math.sin(yaw)) + (Y_p * math.sin(pitch) * math.cos(yaw)) + (Z_p * math.cos(pitch) * math.cos(yaw))
                Y = (X_p * math.cos(yaw)) + (Y_p * math.sin(pitch) * math.sin(yaw)) + (Z_p * math.sin(yaw) * math.cos(pitch))
                #calculate UTM coordinates
                Easting = UTM[3] + Y
                Northing = UTM[4] + X
                Easting = round(Easting,2)
                Northing = round(Northing,2)
				#outputting x and y coords to console
				geocoordinates.append([Easting,Northing])
				#debug.append([u,v,pitch,yaw,UTM[3],UTM[4],X_p,Y_p,X,Y,Easting,Northing])
	
	# Checkpoint
	print('Finsihed processing '+str(n+1)+' of '+str(len(dir_list))+' images.')

# Label coordinates
geocoordinates = np.array(geocoordinates)
df1 = pd.DataFrame({'Easting':geocoordinates[:,0]})
df2 = pd.DataFrame({'Northing':geocoordinates[:,1]})
df3 = pd.DataFrame({'Image':image_name})
output = pd.concat([df1.Easting, df2.Northing, df3.Image], axis=1)

#Save calculated geocoordinates to csv file
#np.savetxt(directory_path+'/image_names.csv',image_name, fmt='%s', header='Image')
#np.savetxt(directory_path+'/geocoordinates.csv',geocoordinates, fmt = '%.2f,%.2f', header = 'Easting, Northing')
#np.savetxt(directory_path+'/debug.csv', debug, fmt = '%.2f', delimiter = ',', header = 'u, v, pitch, yaw, AC_X, AC_Y, X_P, Y_P, X, Y, Easting, Northing')
output.to_csv(directory_path+'/output.csv',index=False)
print('Finished processing directory.')