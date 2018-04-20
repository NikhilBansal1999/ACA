from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import pickle
import numpy
from keras import optimizers

feature_file=open('features.pkl','rb')
features_dict=pickle.load(feature_file)
feature_file.close()

des_file=open('descriptions.pkl','rb')
des_dict=pickle.load(des_file)
des_file.close()

token_file=open('tokens.pkl','rb')
token=pickle.load(token_file)
token_file.close()

max_len=28
vocab_size=4484
def data():
	global features_dict
	global des_dict
	global token
	while 1:
		for key in features_dict:
			im_data=features_dict[key]
			im_data=im_data.flatten()
			im_desc=des_dict[key]
			#input to neural net will consists of im_data concated with word and output will be the next word
			X_train=list()
			Y_train=list()
			im_desc=im_desc.split()
			for i in range(len(im_desc)-1):
				word_append=[token[im_desc[i]]]
				word_out=[token[im_desc[i+1]]]
				word_append=keras.utils.to_categorical(word_append, num_classes=4484)[0]
				word_out=keras.utils.to_categorical(word_out, num_classes=4484)[0]
				if len(numpy.concatenate((im_data,word_append)))==29572:
					X_train.append(numpy.concatenate((im_data,word_append)))
					Y_train.append(word_out)
			X_train=numpy.array(X_train)
			Y_train=numpy.array(Y_train)
			if len(X_train)>0:
				yield (X_train,Y_train)
		
ep=open('epochs.txt','r')
epo=int(ep.read())
ep.close()

model=Sequential()
model.add(Dense(200,input_dim=29572,activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(4484,activation='softmax'))

sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('my_model_weights.h5')
train_generator=data()
model.fit_generator(generator=train_generator,steps_per_epoch=len(features_dict),epochs=1)
model.save_weights('my_model_weights.h5')

ep=open('epochs.txt','w')
ep.write(epo+1)
ep.close()
			
			
		
		