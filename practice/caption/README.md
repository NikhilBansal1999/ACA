# Image-Captioning

The directory contains all the files included as a part of image-captioning project.

The code calc_features.py uses the pre-trained VGG16 model to extract key features 
out of an image and store it in a python dict with keys as the corresponding image name
The dict is then saved as festures.pkl file to be used later.

clean_desc.py reades the description of image from the given file ,cleans it(removes punctuation) and 
stores the cleaned description in description.txt file

pickle_desc.py saves the description of images in a python dict as a pickle file.

train.py trains an  image-captioning model using the features of photo and 
desription generated from the data