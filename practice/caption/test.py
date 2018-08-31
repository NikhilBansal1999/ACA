from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
import pickle
import numpy
from keras import optimizers

feature_file=open('features.pkl','rb')
features_dict=pickle.load(feature_file)
feature_file.close()

token_file=open('new_tokens.pkl','rb')
token=pickle.load(token_file)#key are words
token_file.close()

index_list=dict()
for key in token:
	index_list[str(int(token[key]))]=key
	
sentence_tokens=[]
sentence_tokens.append(token["<start_desc>"])

model=Sequential()
model.add(Dense(512,input_dim=29588,activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(4500,activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.load_weights('my_model_weights.h5')

for key in features_dict:
	im_data=features_dict[key]/100
	im_data=im_data.flatten()
	print(key)
	prev_word_appended=keras.utils.to_categorical([sentence_tokens[-1]], num_classes=4500)[0]
	#print(prev_word_appended.sum())
	for i in range(20):
		word_append=keras.utils.to_categorical([sentence_tokens[-1]], num_classes=4500)[0]
		prev_word_appended=prev_word_appended+word_append
		#print(prev_word_appended.max())
		if len(numpy.concatenate((im_data,prev_word_appended)))==29588:
			X=list()
			X.append(numpy.concatenate((im_data,word_append)))
			X=numpy.array(X)
			next_word=model.predict(X).flatten()
			sentence_tokens.append(next_word.argmax())
			#print(next_word.argmax())
	break
	
for w in sentence_tokens:
	try:
		print(index_list[str(int(w))],end=" ")
	except:
		continue