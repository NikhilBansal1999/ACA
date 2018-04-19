import pickle
word=list()#list of all word in captions
des_file=open("descriptions.txt",'r')
max_len=0
for line in des_file:
	des=line.split()[1:]
	max_len=max([max_len,len(des)])
	for w in des:
		if w not in word:
			word.append(w)
			
word_dict=dict()
i=0
for w in word:
	word_dict[w]=i
	i=i+1
	
out_file=open('tokens.pkl','wb')
pickle.dump(word_dict,out_file)
out_file.close()
des_file.close()
	