import torch
from ResNet_Blocks_3D import resnet18
import torch.nn as nn
import torchvision.transforms as transforms
import os
from CustomDataset_real_gray_images import CustomImageDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 12, activation='leaky_relu').to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('resnet_pose_220626.pt'))

date = '220629'
batch_size = 200
transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])

crop_coor_folder = '../lookup_table_head/training_data_3D_pose_shifted_1/annotations_' + date + '_pose_tensor/'
crop_coor_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_coor_files]

real_folder_add = '../validation_data_3D_pose_shifted/images_real/'
gray_folder_add = '../validation_data_3D_pose_shifted/images_gray/'

im_files = sorted(os.listdir(real_folder_add))

data = CustomImageDataset(im_files,  real_folder_add, gray_folder_add, transform=transform)

#loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=4,prefetch_factor=2,persistent_workers=True)

loader = DataLoader(data, batch_size=batch_size,shuffle=False)

named_layers = dict(model.named_modules())

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


feature_names = {}
n_layers = 4

feature_names[0] = "gate"
feature_names[1] = "block 0"
feature_names[2] = "block 1"
feature_names[3] = "block 2"

model._modules['module'].encoder.gate[0].register_forward_hook(get_features(feature_names[0]))
for i in range(1,4):
    model._modules['module'].encoder.blocks[i-1].blocks[1].blocks[2].register_forward_hook(get_features(feature_names[i]))

PREDS_real = {}
FEATS_real = {}
for i in range(0, n_layers):
    FEATS_real[i] = []
PREDS_gray = {}
FEATS_gray = {}
for i in range(0, n_layers):
    FEATS_gray[i] = []
loss = []
criterion = nn.MSELoss(reduction='mean')

# placeholder for batch features
features = {}
loss = [[],[],[],[]]
#print(model)
# loop through batches
for idx, data in enumerate(loader):
    print(str(idx) + ' of ' + str(len(loader)))
    im_three_channels_real, im_three_channels_gray = data 

    # move to device
    im_three_channels_real = im_three_channels_real.to(device)
    im_three_channels_gray = im_three_channels_gray.to(device)
    
    for layer in range(0,n_layers):
        # Real data
        preds_real = model(im_three_channels_real)
        #PREDS_real.append(preds_real[0].detach().cpu().numpy())
        #FEATS_real.append(features[feature_names[layer]].cpu().numpy())
        f_real = features[feature_names[layer]]
        f_real = torch.flatten(f_real, 1, 3).detach().cpu().numpy()
        FEATS_real[layer].append(f_real)
        
        # Gray data
        preds_gray = model(im_three_channels_gray)
        #PREDS_gray.append(preds_gray[0].detach().cpu().numpy())
        f_gray = features[feature_names[layer]]
        f_gray = torch.flatten(f_gray, 1, 3).detach().cpu().numpy()
        FEATS_gray[layer].append(f_gray)
        #loss[layer].append(criterion(f_real, f_gray))
    # early stop
    #if idx == 30:
    #    break

#PREDS_real = np.concatenate(PREDS_real)

#PREDS_gray = np.concatenate(PREDS_gray)

for i in range(0, n_layers):
    FEATS_real[i] = np.concatenate(FEATS_real[i])
    FEATS_gray[i] = np.concatenate(FEATS_gray[i])

# TSNE
tsne = TSNE(n_components=2, verbose=1, learning_rate = 30.0, n_iter=3000, perplexity=20.0)

n_real = FEATS_real[0].shape[0]
n_gray = n_real
subplot_array = [141,142,143,144]
for i in range(0,n_layers):
    z = tsne.fit_transform(np.concatenate((FEATS_real[i],FEATS_gray[i])))
    plt.subplot(subplot_array[i])
    plt.scatter(z[:n_real,0], z[:n_real,1], s=0.75, c='red')
    plt.scatter(z[n_real:,0], z[n_real:,1], s=0.75, c='green')
    np.save('outputs/tSNE_vector_' + str(i+1) + '_layer.npy', z)
plt.savefig('outputs/layers.png')
print(n_real)
