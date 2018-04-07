import sys
import os
import numpy as np
from PIL import Image
from operator import itemgetter
import time
#achieved an accuracy of about 80%

path = "train"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
# print(folder_names)

# Each row is an image
img = np.zeros([28000, 2025], dtype = float)
labels = np.zeros([28000],dtype=int)

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten()
		img[j] = img[j] / 255.0
		labels[j] = folder_names[i]
		j += 1

test_img = np.zeros([1000, 2025], dtype = float)
test_labels = np.zeros([1000],dtype=int)
path = "test"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		test_img[j] = np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten()
		test_img[j] = test_img[j] / 255.0
		test_labels[j] = folder_names[i]
		j += 1

correct,total=0,0#To keep track of algorithms accuracy
for i in range(1000):
	distance=list()#stores image_label and its distance 
	#compute distance of image i from all train_images
	for k in range(28000):
		dis=np.sqrt(np.sum((img[k]-test_img[i])**2))#calculating distance
		distance.append([dis,labels[k]])
	#sorting distance list() based on distance
	distance=sorted(distance)
	#using 15 nearest neighbours
	distance=distance[:15]
	#storing the number of times each digit appears in closest 7 neighbour
	count=[0,0,0,0,0,0,0,0,0,0]
	for k in range(15):
		count[distance[k][1]]=count[distance[k][1]]+1/distance[k][0]
	#searching for digit appearing most frequently in 15 nearest neighbours
	digit=count.index(max(count))
	"""for k in range(1,10):
		if count[k]>max_count:
			max_count=count[k]
			digit=k"""
	total=total+1
	if digit==test_labels[i]:
		correct+=1
	print(digit,test_labels[i],correct,total)
		

		
	