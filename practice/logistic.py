#python3
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
img = np.zeros([160, 2026], dtype = float)
labels = np.zeros([160],dtype=int)

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.concatenate((np.array([1]),(np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten())/255.0))
		labels[j] = folder_names[i]
		j += 1

test_img = np.zeros([40, 2026], dtype = float)
test_labels = np.zeros([40],dtype=int)
path = "test"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		test_img[j] = np.concatenate((np.array([1]),(np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten())/255.0))
		test_labels[j] = folder_names[i]
		j += 1

correct,wrong=0,0#To keep track of algorithms accuracy
parameters=np.zeros((10,2026))

def prediction(image,digit):#provide image as a vector of length 2026 with first element as one
	global parameters
	hox=np.sum(parameters[digit]*image)
	pre_prob=1/(1+np.exp(hox))#returns the sigmoid of theta*image
	return pre_prob
	
def cost(digit):#provide image as a vector of length 2026 with first element as one
	total_cost=0
	global labels
	global img
	label=np.array(labels==digit,dtype=int)
	for i in range(160):
		predicted=prediction(img[i],digit)#get predicted value
		co = -label[i]*np.log(predicted)-(1-label[i])*np.log(1-prediction)#computes cost
		total_cost=total_cost+co
	return total_cost/160#returns total cost
	
def gradient(digit):
	global parameters
	global img
	global labels
	label=np.array(labels==digit,dtype=int)
	pred=np.array([prediction(im,digit) for im in img])#get predicted value for all digit
	diff=pred-label
	grad=np.dot(diff,img)#a 1*2026 matrix containing gradient with respect to all parameters
	return grad/160
	
#Train Algorithm
for i in range(10):#train algorithms for each digit
	for j in range(1000):#run gradient descend for 1000 iterations for each digit
		grad_desc=gradient(i)
		parameters[i]=parameters[i]-0.01*grad_desc#learning rate = 0.01
		
#Test Algorithm
#print(parameters[1])
#print()
#print()
#print()
correct,wrong=0,0
for i in range(40):
	pred_list=np.array([0 for i in range(10)])#stores prediction probablities
	for j in range(10):
		pred_list[j]=prediction(test_img[i],j)
	pred_digit=np.argmax(pred_list)
	print("Predicted Digit",pred_digit)
	print("Actual Digit",test_labels[i])
	print()
	if pred_digit==test_labels[i]:
		correct=correct+1
	else:
		wrong=wrong+1
		
print("Correct Predictions",correct)
print("Wrong Predictions",wrong)

	
	