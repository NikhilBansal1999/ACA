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

print(type(features_dict))
print(type(des_dict))
print(type(token))
i=0
for key in features_dict:
	print(type(features_dict[key]))
	print(type(des_dict[key]))
	print(des_dict[key])
	print(features_dict[key].shape)
	im_data=features_dict[key].flatten()
	for j in im_data:
		print(j)
	i=i+1
	if i>0:
		break