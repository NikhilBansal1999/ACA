#python3
#obtained an accuracy of 91.5%
import sys
import os
import numpy as np
from PIL import Image
from operator import itemgetter
import time
from sklearn.ensemble import RandomForestClassifier

path = "train"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
# print(folder_names)

# Each row is an image
img = np.zeros([28000, 2026], dtype = float)
labels = np.zeros([28000],dtype=int)

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		img[j] = np.concatenate((np.array([1]),(np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten())/255.0))
		labels[j] = folder_names[i]
		j += 1

test_img = np.zeros([1000, 2026], dtype = float)
test_labels = np.zeros([1000],dtype=int)
path = "test"
folder_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

j = 0
for i in range(10):
	for image in os.listdir(path + "/" + folder_names[i]):
		train_image = path + "/" + folder_names[i] + "/" + image
		test_img[j] = np.concatenate((np.array([1]),(np.asarray(Image.open(train_image).convert('L').resize((45,45), Image.ANTIALIAS)).flatten())/255.0))
		test_labels[j] = folder_names[i]
		j += 1
		
model=RandomForestClassifier(n_estimators=50,criterion='gini',warm_start=False)
pred=model.fit(img,labels)
result=pred.score(test_img,test_labels)
print(result)