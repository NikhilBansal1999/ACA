#The file loads descriptions of an image and stores the first description of the image after cleaning it
#punctuations and single lettered words are removed and all words are converted to lower case
import string
inp_file=open('Flickr8k.token.txt','r')
desc=dict()
table=str.maketrans('','',string.punctuation)
for line in inp_file:
	im_name,im_desc=line.split()[0],line.split()[1:]
	im_name=im_name.split('.')[0]
	des=' '.join(im_desc)
	des=des.translate(table)
	des=des.split()
	describe=list()
	for word in des:
		if len(word)>1:
			describe.append(word.lower())
	describe=' '.join(describe)
	if im_name not in desc:
		desc[im_name]=describe
		
out_file=('description.txt')
text=list()
for key in desc:
	text.append(key+' '+desc[key])
out_file=open('descriptions.txt','w')
out_file.write('\n'.join(text))
inp_file.close()
out_file.close()
	