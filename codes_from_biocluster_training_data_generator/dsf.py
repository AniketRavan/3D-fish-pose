f = open('runme_vids_models_free.m','r')
text = f.readlines()
f.close()
n = 80;
temp = text[9][13:];
for i in range(0,n):
	temp = text[9][0:12] + 'floor(length(mats)*'+ str(i)+'/'+str(n)+' + 1):floor(length(mats)*'+str(i+1)+'/'+str(n)+')'
	text[9] = temp
	filename = 'runme_vids_models_free_'+str(i+1)+'er.m'
	with open(filename,'w') as f:
		f.writelines(text)
