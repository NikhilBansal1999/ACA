import pickle
in_file=open('descriptions.txt','r')
des_dict=dict()
i=0
for line in in_file:
	key,desc=line.split()[0],' '.join(line.split()[1:])
	des_dict[key]=desc
	i=i+1
	
out_file=open('descriptions.pkl','wb')
pickle.dump(des_dict,out_file)
out_file.close()
in_file.close()
print(i)
	