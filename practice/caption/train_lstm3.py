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

os.environ["CUDA_VISIBLE_DEVICES"]="3"

feature_file=open('features.pkl','rb')
features_dict=pickle.load(feature_file)
feature_file.close()

des_file=open('new_descriptions.pkl','rb')
des_dict=pickle.load(des_file)
des_file.close()

token_file=open('new_tokens.pkl','rb')
token=pickle.load(token_file)
token_file.close()

max_len=20
vocab_size=4500
def data():
	global features_dict
	global des_dict
	global token
	while 1:
		for key in features_dict:
			im_data=features_dict[key]/100
			im_data=im_data.squeeze().flatten()
			im_desc=des_dict[key]
			#input to neural net will consists of im_data and partial captions
			im_desc=im_desc.split()
			next_word=[]
			incomplete_captions=[]
			current_image=[]
			num_appended=0

			for i in range(0,len(im_desc)-1):
				if num_appended==20: #Maximum caption length is 20
					break
				caps=token[im_desc[i]]
				nextw=numpy.zeros(4500)
				nextw[token[im_desc[i+1]]]=1
				incomplete_captions.append(caps)
				next_word.append(nextw)
				current_image.append(im_data)
				num_appended=num_appended+1
				
			while num_appended<20:
				num_appended=num_appended+1
				caps=token['<end_desc>']
				nextw=numpy.zeros(4500)
				nextw[token['<end_desc>']]=1
				incomplete_captions.append(caps)
				next_word.append(nextw)
				current_image.append(im_data)

			next_word=numpy.array(next_word)
			next_word=numpy.expand_dims(next_word,axis=0)
			current_image=numpy.array(current_image)
			current_image=numpy.expand_dims(current_image,axis=0)
			incomplete_captions=numpy.array(incomplete_captions)
			incomplete_captions=numpy.expand_dims(incomplete_captions,axis=0)
			yield [[current_image,incomplete_captions],next_word]


ep=open('lstm3_epochs.txt','r')
epo=int(ep.read())
epoc=9
ep.close()

image_input=Input(shape=(20,25088))
image_processor_model=Dense(512,activation='relu')(image_input)

part_caps=Input(shape=(20,))
caption_processor=Embedding(4500,256,input_length=20)(part_caps)
caption_processor=LSTM(256,return_sequences=True)(caption_processor)
caption_processor=TimeDistributed(Dense(512))(caption_processor)

merge_models=add([image_processor_model,caption_processor])
caption_generator=Dense(4500,activation='softmax')(merge_models)

Captioning_model=Model(inputs=[image_input,part_caps], outputs=caption_generator)
optimize=Adam(lr=0.01)
Captioning_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
Captioning_model.load_weights('my_model_weights_lstm3_model_v1.h5')
print(Captioning_model.summary())

train_generator=data()
Captioning_model.fit_generator(generator=train_generator,steps_per_epoch=len(features_dict),epochs=epoc)
Captioning_model.save_weights('my_model_weights_lstm3_model_v1.h5')

ep=open('lstm3_epochs.txt','w')
ep.write(str(epo+epoc))
ep.close()
