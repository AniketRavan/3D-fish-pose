from os import listdir
import sys

f = open('runme_generate_training_data.m','r')
text = f.readlines()
f.close()
temp = text[27];
n_files = 100
data_size = 500000
batch_size = int(data_size/n_files)
for i in range(0,n_files):
	filename = 'parallel_data_generation/runme_' + str(i+1) + '.m'
	max_idx = (i+1)*batch_size
	idx = i*batch_size
	temp = 'idx = ' + str(idx) + '\n'
	text[4] = "load('../proj_params_101019_corrected_new')" + '\n'
	text[5] = "load('../lut_b_tail')" + '\n'
	text[6] = "load('../lut_s_tail')" + '\n'
	text[7] = "path{1} = '../../results_all_er';" + '\n'
	text[10] = 'idx = ' + str(idx) + ';' + '\n'
	text[11] = "data_dir = '../../training_data_3D_pose_shifted/'" + '\n'
	text[88] = '                    if (mod(idx,1) == ' + str(batch_size) + '); display(idx); end' + '\n'
	text[89] = '                    if idx == ' + str(max_idx) + '\n' 
	text[90] = "                        display('Finished ' + num2str(" + str(max_idx) + ")  )" + '\n'
	text[91] = '                        return' + '\n'
	text[92] = '                    end' + '\n'
	with open(filename,'w') as f:
		f.writelines(text)
