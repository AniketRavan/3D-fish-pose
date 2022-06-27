import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from CustomDataset import CustomImageDataset
from ResNet_Blocks_3D import resnet18
import time
from multiprocessing import Pool
import os
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=10, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="../outputs_resnet", type=str, help='path to store output images and plots')
parser.add_argument('-p','--proj_params', default="proj_params_101019_corrected_new", type=str, help='path to calibrated camera parameters')

imageSizeX = 141
imageSizeY = 141

args = vars(parser.parse_args())

proj_params_path = args['proj_params']
epochs = args['epochs']
output_dir = args['output_dir']
proj_params = sio.loadmat(proj_params_path)
proj_params = torch.tensor(proj_params['proj_params'])
proj_params = proj_params[None,:,:]
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 10, activation='leaky_relu').to(device)
n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 1500*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

model.load_state_dict(torch.load('resnet_pose_220621.pt'))

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
im_folder = '../../validation_data_3D_pose_new/images/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

crop_coor_folder = '../../validation_data_3D_pose_new/annotations_220623_crop_coor_tensor/'
crop_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_files]


val_data = CustomImageDataset(im_files_add, [], [], crop_coor_files_add, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')

def calc_proj_w_refra(coor_3d, proj_params):
    fa1p00 = proj_params[0,0,0]
    fa1p10 = proj_params[0,0,1]
    fa1p01 = proj_params[0,0,2]
    fa1p20 = proj_params[0,0,3]
    fa1p11 = proj_params[0,0,4]
    fa1p30 = proj_params[0,0,5]
    fa1p21 = proj_params[0,0,6]
    fa2p00 = proj_params[0,1,0]
    fa2p10 = proj_params[0,1,1]
    fa2p01 = proj_params[0,1,2]
    fa2p20 = proj_params[0,1,3]
    fa2p11 = proj_params[0,1,4]
    fa2p30 = proj_params[0,1,5]
    fa2p21 = proj_params[0,1,6]
    fb1p00 = proj_params[0,2,0]
    fb1p10 = proj_params[0,2,1]
    fb1p01 = proj_params[0,2,2]
    fb1p20 = proj_params[0,2,3]
    fb1p11 = proj_params[0,2,4]
    fb1p30 = proj_params[0,2,5]
    fb1p21 = proj_params[0,2,6]
    fb2p00 = proj_params[0,3,0]
    fb2p10 = proj_params[0,3,1]
    fb2p01 = proj_params[0,3,2]
    fb2p20 = proj_params[0,3,3]
    fb2p11 = proj_params[0,3,4]
    fb2p30 = proj_params[0,3,5]
    fb2p21 = proj_params[0,3,6]
    fc1p00 = proj_params[0,4,0]
    fc1p10 = proj_params[0,4,1]
    fc1p01 = proj_params[0,4,2]
    fc1p20 = proj_params[0,4,3]
    fc1p11 = proj_params[0,4,4]
    fc1p30 = proj_params[0,4,5]
    fc1p21 = proj_params[0,4,6]
    fc2p00 = proj_params[0,5,0]
    fc2p10 = proj_params[0,5,1]
    fc2p01 = proj_params[0,5,2]
    fc2p20 = proj_params[0,5,3]
    fc2p11 = proj_params[0,5,4]
    fc2p30 = proj_params[0,5,5]
    fc2p21 = proj_params[0,5,6]
    npts = coor_3d.shape[2]
    coor_b = torch.zeros(coor_3d.shape[0],2,npts)
    coor_s1 = torch.zeros(coor_3d.shape[0],2,npts)
    coor_s2 = torch.zeros(coor_3d.shape[0],2,npts)
    coor_b[:,0,:] = fa1p00+fa1p10*coor_3d[:,2,:]+fa1p01*coor_3d[:,0,:]+fa1p20*coor_3d[:,2,:]**2+fa1p11*coor_3d[:,2,:]*coor_3d[:,0,:]+fa1p30*coor_3d[:,2,:]**3+fa1p21*coor_3d[:,2,:]**2*coor_3d[:,0,:]
    coor_b[:,1,:] = fa2p00+fa2p10*coor_3d[:,2,:]+fa2p01*coor_3d[:,1,:]+fa2p20*coor_3d[:,2,:]**2+fa2p11*coor_3d[:,2,:]*coor_3d[:,1,:]+fa2p30*coor_3d[:,2,:]**3+fa2p21*coor_3d[:,2,:]**2*coor_3d[:,1,:]
    coor_s1[:,0,:] = fb1p00+fb1p10*coor_3d[:,0,:]+fb1p01*coor_3d[:,1,:]+fb1p20*coor_3d[:,0,:]**2+fb1p11*coor_3d[:,0,:]*coor_3d[:,1,:]+fb1p30*coor_3d[:,0,:]**3+fb1p21*coor_3d[:,0,:]**2*coor_3d[:,1,:]
    coor_s1[:,1,:] = fb2p00+fb2p10*coor_3d[:,0,:]+fb2p01*coor_3d[:,2,:]+fb2p20*coor_3d[:,0,:]**2+fb2p11*coor_3d[:,0,:]*coor_3d[:,2,:]+fb2p30*coor_3d[:,0,:]**3+fb2p21*coor_3d[:,0,:]**2*coor_3d[:,2,:]
    coor_s2[:,0,:] = fc1p00+fc1p10*coor_3d[:,1,:]+fc1p01*coor_3d[:,0,:]+fc1p20*coor_3d[:,1,:]**2+fc1p11*coor_3d[:,1,:]*coor_3d[:,0,:]+fc1p30*coor_3d[:,1,:]**3+fc1p21*coor_3d[:,1,:]**2*coor_3d[:,0,:]
    coor_s2[:,1,:] = fc2p00+fc2p10*coor_3d[:,1,:]+fc2p01*coor_3d[:,2,:]+fc2p20*coor_3d[:,1,:]**2+fc2p11*coor_3d[:,1,:]*coor_3d[:,2,:]+fc2p30*coor_3d[:,1,:]**3+fc2p21*coor_3d[:,1,:]**2*coor_3d[:,2,:]
    return coor_b - 1, coor_s1 - 1, coor_s2 - 1 # Subtract 1 to abide with MATLAB's indices

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def validate(model, dataloader, proj_params):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        #for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        for i, data in enumerate(dataloader):
            for p in range(0,1): 
                im_three_channels, crop_coor_data = data
                im_three_channels = im_three_channels.to(device)
                crop_coor_data = crop_coor_data.to(device)
                crop_split = crop_coor_data[:,0,[0,2,4,6,8,10]]
                proj_params = proj_params.to(device)
                coor_3d = model(im_three_channels, crop_split)
                pose_recon_b, pose_recon_s1, pose_recon_s2 = calc_proj_w_refra(coor_3d, proj_params)
                pose_recon_b = pose_recon_b.to(crop_coor_data.device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s1 = pose_recon_s1.to(crop_coor_data.device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s2 = pose_recon_s2.to(crop_coor_data.device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_b[:,0,:] = pose_recon_b[:,0,:] - crop_coor_data[:,0,2,None] 
                pose_recon_b[:,1,:] = pose_recon_b[:,1,:] - crop_coor_data[:,0,0,None] 
                pose_recon_s1[:,0,:] = pose_recon_s1[:,0,:] - crop_coor_data[:,0,6,None] 
                pose_recon_s1[:,1,:] = pose_recon_s1[:,1,:] - crop_coor_data[:,0,4,None] 
                pose_recon_s2[:,0,:] = pose_recon_s2[:,0,:] - crop_coor_data[:,0,10,None] 
                pose_recon_s2[:,1,:] = pose_recon_s2[:,1,:] - crop_coor_data[:,0,8,None] 
            #    pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,20:30])
            #    eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
                #pose_loss = criterion(coor_3d_data[:,:,0:10], coor_3d[:,:,0:10])
            #    eye_loss = criterion(coor_3d_data[:,:,10:12], coor_3d[:,:,10:12])
                #loss = pose_loss# + eye_loss
                #running_loss += loss.item()
                #running_pose_loss += pose_loss.item()
                
                # save the last batch input and output of every epoch
                if i == int(len(val_data)/dataloader.batch_size) - 1:
                    num_rows = 8
                    im_b = im_three_channels[:,0,:,:]
                    im_s1 = im_three_channels[:,1,:,:]
                    im_s2 = im_three_channels[:,2,:,:]
                    images_b = im_b.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                    images_s1 = im_s1.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                    images_s2 = im_s2.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                    _,axs = plt.subplots(nrows=6, ncols=8)

                    # Overlay pose
                    for m in range(0,8):
                        #print(coor_3d[m,:,:])
                        axs[1,m].imshow(images_b[m,0,:,:].cpu(), cmap='gray')
                        axs[1,m].scatter(pose_recon_b[m,0,:].cpu(), pose_recon_b[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[0,m].imshow(images_b[m,0,:,:].cpu(), cmap='gray')
                #        axs[0,m].scatter(pose_data[m,0,0:10].cpu(), pose_data[m,1,0:10].cpu(), s=0.07, c='red', alpha=0.6)
             #          axs[0,m].scatter(eye_coor_data[m,0,0:2].cpu(), eye_coor_data[m,1,0:2].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[3,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                        axs[3,m].scatter(pose_recon_s1[m,0,:].cpu(), pose_recon_s1[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[2,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                 #       axs[2,m].scatter(pose_data[m,0,10:20].cpu(), pose_data[m,1,10:20].cpu(), s=0.07, c='red', alpha=0.6)
              #         axs[2,m].scatter(eye_coor_data[m,0,2:4].cpu(), eye_coor_data[m,1,2:4].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[5,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                        axs[5,m].scatter(pose_recon_s2[m,0,:].cpu(), pose_recon_s2[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[4,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                  #      axs[4,m].scatter(pose_data[m,0,20:30].cpu(), pose_data[m,1,20:30].cpu(), s=0.07, c='red', alpha=0.6)
               #        axs[4,m].scatter(eye_coor_data[m,0,4:6].cpu(), eye_coor_data[m,1,4:6].cpu(), s=0.07, c='green', alpha=0.6)



                    plt.savefig(output_dir + "/epoch_" + str(epoch) + ".svg")
                    plt.close()

    val_loss = 0
    return val_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    val_epoch_loss = validate(model, val_loader, proj_params)
    #val_loss.append(val_epoch_loss)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)


