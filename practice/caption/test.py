from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import pickle
import numpy
from keras import optimizers

feature_file=open('features.pkl','rb')
features_dict=pickle.load(feature_file)
feature_file.close()

token_file=open('tokens.pkl','rb')
token=pickle.load(token_file)#key are words
token_file.close()

index_list=dict()
for key in token:
	index_list[str(int(token[key]))]=key
	
sentence_tokens=[]
sentence_tokens.append(token["the"])
model=Sequential()
model.add(Dense(256,input_dim=29572,activation='relu'))
#model.add(Dense(200, activation='relu'))
model.add(Dense(4484,activation='softmax'))
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.load_weights('my_model_weights.h5')

for key in features_dict:
	im_data=features_dict[key]/100
	im_data=im_data.flatten()
	print(key)
	for i in range(20):
		word_append=keras.utils.to_categorical([sentence_tokens[-1]], num_classes=4484)[0]
		if len(numpy.concatenate((im_data,word_append)))==29572:
			X=list()
			X.append(numpy.concatenate((im_data,word_append)))
			X=numpy.array(X)
			next_word=model.predict(X).flatten()
			sentence_tokens.append(next_word.argmax())
	break
	
for w in sentence_tokens:
	try:
		print(index_list[str(int(w))],end=" ")
	except:
		continue