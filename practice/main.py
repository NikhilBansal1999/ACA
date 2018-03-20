import sys
import os
import numpy as np
from PIL import Image
from operator import itemgetter
import time


path = "train"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
# print(folder_names)

# Each row is an image
img = np.zeros([160, 2025], dtype = float)
labels = np.zeros([160],dtype=int)

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten()
		img[j] = img[j] / 255.0
		labels[j] = folder_names[i]
		j += 1

test_img = np.zeros([40, 2025], dtype = float)
test_labels = np.zeros([40],dtype=int)
path = "test"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		test_img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten()
		test_img[j] = img[j] / 255.0
		test_labels[j] = folder_names[i]
		j += 1

correct,wrong=0,0#To keep track of algorithms accuracy
for i in range(40):
	print(i)
	distance=list()#stores image_label and its distance 
	#compute distance of image i from all train_images
	for k in range(160):
		dis=0
		dis=dis+np.sqrt(np.sum((img[k]-test_img[i])**2))#calculating distance
		distance.append([dis,labels[k]])
	#sorting distance list() based on distance
	distance=sorted(distance)
	#using 15 nearest neighbours
	distance=distance[:15]
	print(distance)
	#storing the number of times each digit appears in closest 7 neighbour
	count=[0,0,0,0,0,0,0,0,0,0]
	print(count)
	for k in range(15):
		count[distance[k][1]]=count[distance[k][1]]+1
	#searching for digit appearing most frequently in 15 nearest neighbours
	print(count)
	digit=0
	max_count=count[0]
	for k in range(1,10):
		if count[k]>max_count:
			max_count=count[k]
			digit=k
	print("Prediction is :",digit)
	print("True Digit is :",test_labels[i])
	print()
	if digit==test_labels[i]:
		correct=correct+1
	else:
		wrong=wrong+1
	time.sleep(5)
		
print("Wrong Predictions:",wrong)
print("Correct Predictions:",correct)
		
	