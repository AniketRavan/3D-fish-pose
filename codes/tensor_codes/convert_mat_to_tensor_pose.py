import scipy.io as sio
import torch
import os
import numpy as np

date='220626'
folder_path = '../annotations_' + date + '_pose'
new_dir = '../annotations_' + date + '_pose_tensor'
files = os.listdir(folder_path)

if (not os.path.isdir(new_dir)):
	os.mkdir(new_dir)

for i in range(0,len(files)):
	if (i%100000):
		print(i/len(files)*100,flush=True)
	data_dir = sio.loadmat(folder_path + '/' + files[i])
	mat = data_dir['pose']
	tensor = torch.tensor(mat,dtype=torch.float32)
	torch.save(tensor, new_dir + '/' + files[i][:-3] + 'pt')
