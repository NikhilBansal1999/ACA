from keras.models import Sequential
from keras.layers import Dense,Conv2D,Embedding,LSTM,concatenate,Input,Reshape
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import keras
import pickle
import numpy
from keras import optimizers

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
			im_data=im_data.squeeze()
			im_desc=des_dict[key]
			#input to neural net will consists of im_data and partial captions
			im_desc=im_desc.split()
			next_word=[]
			incomplete_captions=[]
			current_image=[]
			if len(im_desc) < 4:
				continue

			for i in range(1,len(im_desc)-1):
				if i==21: #Maximum caption length is 20
					break
				caps=[token[word] for word in im_desc[:i]]
				nextw=numpy.zeros(4500)
				nextw[token[im_desc[i]]]=1
				incomplete_captions.append(caps)
				next_word.append(nextw)
				current_image.append(im_data)

			next_word=numpy.array(next_word)
			incomplete_captions=pad_sequences(incomplete_captions,maxlen=20,padding='post')
			current_image=numpy.array(current_image)
			if current_image.ndim is not 4:
				continue

			yield [[current_image,incomplete_captions],next_word]

ep=open('epochs.txt','r')
epo=int(ep.read())
epoc=1
ep.close()

optimize=SGD(lr=0.01)

image_input=Input(shape=(7,7,512))
image_processor_model=Conv2D(64,(3,4))(image_input)
image_processor_model=Dense(512,activation='elu')(image_processor_model)
image_processor_model=Reshape((20,512))(image_processor_model)

part_caps=Input(shape=(20,))
caption_processor=Embedding(4500,128,input_length=20)(part_caps)
caption_processor=LSTM(256,return_sequences=True)(caption_processor)
caption_processor=Dense(512,activation='relu')(caption_processor)

merge_models=concatenate([image_processor_model,caption_processor])
caption_generator=LSTM(256)(merge_models)
caption_generator=Dense(4500,activation='softmax')(caption_generator)

Captioning_model=Model(inputs=[image_input,part_caps], outputs=caption_generator)
Captioning_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Captioning_model.load_weights('my_model_weights.h5')

train_generator=data()
Captioning_model.fit_generator(generator=train_generator,steps_per_epoch=len(features_dict),epochs=epoc)
Captioning_model.save_weights('my_model_weights.h5')

ep=open('epochs.txt','w')
ep.write(str(epo+epoc))
ep.close()
