'''
Name: Object_detection_ortho2.py
By: Roberto Rodriguez
Date created: 3/9/2019
Last updated: 3/9/2019
Description:
This program uses a TensorFlow-trained classifier to perform object detection.
It loads the classifier and orthomosaic data. Then breaks the orthomosaic into segments and saves them in a directry.
It draws boxes and scores around the objects of interest in the image segments.
It performs calculations on the centerpoints of the detection boxes and saves the geocoordinates to a CSV file.

Notes:
3/3/2019 - Debug pixel_coordinate array contains segment, local u,v and global u,v coordinates
'''

# Import packages
import os, sys
import cv2
import numpy as np
import tensorflow as tf
import glob
import math
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from numpy import genfromtxt
from PIL import Image, ExifTags
from osgeo import gdal

# Import custom libraries
from convert_to_degrees_2 import dms_to_degrees
from xmp_read import get_xmp
from WGS84toUTM import WGS84toUTM
from sensor_dim import get_sensor

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

#Adding the image
image_path = askopenfilename(initialdir="C:/",title='Choose the TIF image.')
#image_path = 'C:/Users/Frost-Desktop/Desktop/Coordinate_Workflow/Test_Dir/Arno_Ortho_subsets/Arno_150m_sub1.jpg'

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

# Retreive image information
ds = gdal.Open(image_path)

# Get geographic transofrmation information (returns upper left corner)
ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()

# Checkpoint
print('Loaded image metadata.')

# Convert RGB bands to arrays
print('Converting GeoTiff.')
red = ds.GetRasterBand(1)
green = ds.GetRasterBand(2)
blue = ds.GetRasterBand(3)
arr_red = red.ReadAsArray()
arr_green = green.ReadAsArray()
arr_blue = blue.ReadAsArray()

# Get size of image
[cols,rows] = arr_red.shape

# Generate  file for segmentation
outfile = image_path[0:-4]+'_temp.tif'
outdriver = gdal.GetDriverByName("GTiff")
outdata   = outdriver.Create(str(outfile), rows, cols, 3, gdal.GDT_Byte)

outdata.GetRasterBand(1).WriteArray(arr_red)
outdata.GetRasterBand(2).WriteArray(arr_green)
outdata.GetRasterBand(3).WriteArray(arr_blue)

outdata = None
print('GeoTiff converted.')

src = gdal.Open(outfile)

# Make directory
image_dir = image_path[0:-4]
os.mkdir(image_dir)
print('Segmenting image.')

a=1 # Start counting at 1 to simplify later calculations
b=1
c=0
d=0

while d < 1000 * (cols//1000 + 1):
	n_a = str(a)
	if len(n_a) < 2:
		n_a = '0' + n_a
	n_b = str(b)
	if len(n_b) < 2:
		n_b = '0' + n_b

	translate_options = gdal.TranslateOptions(outputType = gdal.GDT_Byte, srcWin = [c, d, 1000, 1000], format='JPEG')
	segment_name = image_dir+'/'+n_b+n_a+'.jpg'
	temp = gdal.Translate(segment_name, src, options = translate_options)
	temp = None
	
	if(c < 1000 * (rows//1000 + 1)):
		c=c+1000
		a=a+1
	if(c >= 1000 * (rows//1000 + 1)):
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

# Create an array to store coordinates
pixel_coordinates = []

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
	cv2.imwrite(image_list[i],image)
	
	# Progress update
	print('Completed '+str(i+1)+' of '+str(len(image_list))+' segments.')
    
    # Save boxes above the minimum threshold and normalize them
	for j, box in enumerate(np.squeeze(boxes)):
		if(np.squeeze(scores)[j] > THRESHOLD):
			y_coord = (box[0]*im_height + box[2]*im_height) / 2
			x_coord = (box[1]*im_width + box[3]*im_width) / 2
			x_coord = int(round(x_coord))
			y_coord = int(round(y_coord))
			local_frame = int(image_list[i][-8:-4])
			global_x = 1000 * (local_frame%100 - 1) + x_coord
			global_y = 1000 * (local_frame//100 - 1) + y_coord
			coord = [local_frame, x_coord, y_coord, global_x, global_y]
			#coord = [global_x, global_y]
			pixel_coordinates.append(coord)

#Checkpoint
print('Object detection completed.')
print('Calculating geocoordinates.')

# Create array to store final coordinates
geocoordinates = []

# Calculate geocoordinates for each pixel coordinate 
for m in range(len(pixel_coordinates)):
	# Place pixel coordinates into u and v
	pc = pixel_coordinates[m]
	u = pc[3]
	v = pc[4]
	box_x = ulx + (u * xres)
	box_y = uly + (v * yres)
	geocoordinates.append([box_x,box_y])

#Save calculated geocoordinates to csv file
np.savetxt(image_path[:-4]+'_pixelcoordinates.csv',pixel_coordinates, fmt = '%.0f', delimiter = ',', header = 'segment, local_u, local_v, global_u, global_v')
np.savetxt(image_path[:-4]+'_geocoordinates.csv',geocoordinates, fmt = '%.2f', delimiter = ',', header = 'Easting, Northing')
print('Finished processing image.')