from keras.models import Sequential
from keras.layers import Dense,Conv2D,Embedding,LSTM,concatenate,Input,Reshape,TimeDistributed,add
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras
import pickle
import numpy
from keras import optimizers
import os
import sys
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16,preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"]="3"

if len(sys.argv) is not 2:
	print('Usage: python test_lstm.py image_name')
	sys.exit()

########              Take image as input and pre-process the image, pass into VGG model and generate features            ###########
inp=Input(shape=(224,224,3))
model=VGG16(include_top=False,input_tensor=inp)
image_name=sys.argv[1]
image=load_img(image_name,target_size=(224,224))
image=img_to_array(image)
image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
image=preprocess_input(image)
feature=model.predict(image,verbose=0)


#########              Creates model            ##########
image_input=Input(shape=(20,25088))
image_processor_model=Dense(512,activation='relu')(image_input)

part_caps=Input(shape=(20,))
caption_processor=Embedding(4500,256,input_length=20)(part_caps)
caption_processor=LSTM(256,return_sequences=True)(caption_processor)
caption_processor=TimeDistributed(Dense(512))(caption_processor)

merge_models=add([image_processor_model,caption_processor])
caption_generator=Dense(4500,activation='softmax')(merge_models)

Captioning_model=Model(inputs=[image_input,part_caps], outputs=caption_generator)
Captioning_model.load_weights('my_model_weights_lstm3_model_v1.h5')
print(Captioning_model.summary())

#######      Read Token File      #########
token_file=open('new_tokens.pkl','rb')
token=pickle.load(token_file)
token_file.close()

######## Start Generating Caption    #######
predicted_words=[]
predicted_words.append(token['<start_desc>'])

im_data=feature/100
im_data=im_data.squeeze().flatten()
for i in range(1,20):
	next_word=[]
	incomplete_captions=[]
	current_image=[]
	for j in range(0,i):
		caps=predicted_words[j]
		incomplete_captions.append(caps)
		current_image.append(im_data)
		
	for j in range(i,20):
		caps=token['<end_desc>']
		incomplete_captions.append(caps)
		current_image.append(im_data)
		
	current_image=numpy.array(current_image)
	current_image=numpy.expand_dims(current_image,axis=0)
	incomplete_captions=numpy.array(incomplete_captions)
	incomplete_captions=numpy.expand_dims(incomplete_captions,axis=0)
	
	predictions=Captioning_model.predict([current_image,incomplete_captions])
	predictions=predictions[0][i-1]
	predictions=numpy.argmax(predictions)
	predicted_words.append(predictions)
	

reverse_token=dict()
end_token=token['<end_desc>']
for key in token:
	reverse_token[token[key]]=key
	
for index in range(1,len(predicted_words)):
	if predicted_words[index]==end_token:
		break
	print(reverse_token[predicted_words[index]],end=" ")