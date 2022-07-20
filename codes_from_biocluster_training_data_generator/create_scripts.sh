f = open('script_training_data.sh','r')
text = f.readlines()
f.close()
n = 100;
for i in range(0,n):
	text[5] = '#SBATCH -J dat_' + str(i+1) + '\n'
	text[6] = '#SBATCH -o dat_' + str(i+1) + '\n'
	text[8] = 'matlab -nodesktop -r "run('"'"'runme_' + str(i+1) + '.m\');exit;"\n'
	filename = 'parallel_data_generation/script_'+str(i+1)+'.sh'
	with open(filename,'w') as f:
		f.writelines(text)
