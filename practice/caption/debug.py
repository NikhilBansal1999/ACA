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

print(type(features_dict))
print(type(des_dict))
print(type(token))

for key in features_dict:
	print(features_dict[key].squeeze().shape)
	break
	
for key in des_dict:
	print(des_dict[key])
	break
	
for key in token:
	print(token["<start_desc>"])
	break

print(len(token.keys()))