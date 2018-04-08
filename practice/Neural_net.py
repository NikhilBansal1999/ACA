#python3
#obtained 82.5% accuracies
import sys
import os
import numpy as np
from PIL import Image
from operator import itemgetter
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


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
		img[j] = img[j]/255.0
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
		test_img[j] = test_img[j]/255.0
		test_labels[j] = folder_names[i]
		j += 1
		
print(labels.shape)
print(labels[0])
print(labels[2900])
labels=keras.utils.to_categorical(labels,10)
test_labels=keras.utils.to_categorical(test_labels,10)
print(labels.shape)
print(labels[0])
print(labels[2900])

model=Sequential()
model.add(Dense(512,input_dim=2025,activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(img,labels,epochs=20,batch_size=140,validation_data=(test_img, test_labels))
score=model.evaluate(test_img,test_labels,batch_size=1000)
print(score)
