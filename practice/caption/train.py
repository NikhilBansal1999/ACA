from keras.models import Sequential
from keras.layers import Dense, Dropout
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
			im_data=im_data.flatten()
			im_desc=des_dict[key]
			#input to neural net will consists of im_data concated with all the previous word and output will be the next word
			X_train=list()
			Y_train=list()
			im_desc=im_desc.split()
			prev_word_appended=keras.utils.to_categorical(0, num_classes=4500)[0]
			for i in range(len(im_desc)-1):
				word_append=[token[im_desc[i]]]
				word_out=[token[im_desc[i+1]]]
				word_append=keras.utils.to_categorical(word_append, num_classes=4500)[0]
				prev_word_appended=prev_word_appended+word_append
				word_out=keras.utils.to_categorical(word_out, num_classes=4500)[0]
				if len(numpy.concatenate((im_data,prev_word_appended)))==29588:
					X_train.append(numpy.concatenate((im_data,prev_word_appended)))
					Y_train.append(word_out)
			X_train=numpy.array(X_train)
			Y_train=numpy.array(Y_train)
			if len(X_train)>0:
				yield (X_train,Y_train)
		
ep=open('epochs.txt','r')
epo=int(ep.read())
epoc=10
ep.close()

model=Sequential()
model.add(Dense(512,input_dim=29588,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(4500,activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('my_model_weights.h5')
train_generator=data()
model.fit_generator(generator=train_generator,steps_per_epoch=len(features_dict),epochs=epoc)
model.save_weights('my_model_weights.h5')

ep=open('epochs.txt','w')
ep.write(str(epo+epoc))
ep.close()
			
			
		
		
