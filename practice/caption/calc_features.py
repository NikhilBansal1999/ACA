from os import listdir
import time
from pickle import dump
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.layers import Input

inp=Input(shape=(224,224,3))
model=VGG16(include_top=False,input_tensor=inp)
features=dict()
for file in listdir('Flicker8k_Dataset'):
	start=time.time()
	image_name='Flicker8k_Dataset/'+file
	image=load_img(image_name,target_size=(224,224))
	image=img_to_array(image)
	image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
	image=preprocess_input(image)
	feature=model.predict(image,verbose=0)
	features[file.split('.')[0]]=feature
	print(time.time()-start)
	
dump(features, open('features.pkl', 'wb'))
	

	