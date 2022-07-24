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
from CustomDataset_val import CustomImageDataset
from ResNet_Blocks_3D import resnet18
import time
from multiprocessing import Pool
import os
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20) 
import time

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=1, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="validations", type=str, help='path to store output images and plots')
parser.add_argument('-p','--proj_params', default="proj_params_101019_corrected_new", type=str, help='path to calibrated camera parameters')

imageSizeX = 141
imageSizeY = 141

args = vars(parser.parse_args())
date = '220720'
proj_params_path = args['proj_params']
epochs = args['epochs']
output_dir = args['output_dir']
proj_params = sio.loadmat(proj_params_path)
proj_params = torch.tensor(proj_params['proj_params'])
proj_params = proj_params[None,:,:]
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 12, activation='leaky_relu').to(device)
n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 300*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

model = nn.DataParallel(model)

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

model.load_state_dict(torch.load('resnet_pose_220717_2_lowest_loss.pt'))

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])

crop_coor_folder = '../validation_data_3D_220720_boundary_noise_10000/annotations_'+date+'_crop_coor_tensor/'
crop_coor_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_coor_files]

coor_3d_folder = '../validation_data_3D_220720_boundary_noise_10000/annotations_'+date+'_coor_3d_tensor/'
coor_3d_files = sorted(os.listdir(coor_3d_folder))
coor_3d_files_add = [coor_3d_folder + file_name for file_name in coor_3d_files]

im_folder = '../validation_data_3D_220720_boundary_noise_10000/images_real/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

val_data = CustomImageDataset(im_files_add, crop_coor_files_add, coor_3d_files_add, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='none')

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

def display_fits(im_b, im_s1, im_s2, pose_data_b, pose_data_s1, pose_data_s2, pose_recon_b, pose_recon_s1, pose_recon_s2, counter, image_dir):
    for i in range(0,im_b.shape[0]):
        _,axs = plt.subplots(nrows=3, ncols=2)
        axs[0,0].imshow(im_b[i,:,:].cpu(), cmap='gray')
        axs[0,0].scatter(pose_data_b[i,0,:].cpu(), pose_data_b[i,1,:].cpu(), s=0.07, c='red', alpha=0.6)
        axs[0,0].grid(False)
        axs[0,0].set_axis_off()

        axs[0,1].imshow(im_b[i,:,:].cpu(), cmap='gray')
        axs[0,1].scatter(pose_recon_b[i,0,:].cpu(), pose_recon_b[i,1,:].cpu(), s=0.07, c='green', alpha=0.6)
        axs[0,1].grid(False)
        axs[0,1].set_axis_off()

        axs[1,0].imshow(im_s1[i,:,:].cpu(), cmap='gray')
        axs[1,0].scatter(pose_data_s1[i,0,:].cpu(), pose_data_s1[i,1,:].cpu(), s=0.07, c='red', alpha=0.6)
        axs[1,0].grid(False)
        axs[1,0].set_axis_off()

        axs[1,1].imshow(im_s1[i,:,:].cpu(), cmap='gray')
        axs[1,1].scatter(pose_recon_s1[i,0,:].cpu(), pose_recon_s1[i,1,:].cpu(), s=0.07, c='green', alpha=0.6)
        axs[1,1].grid(False)
        axs[1,1].set_axis_off()

        axs[2,0].imshow(im_s2[i,:,:].cpu(), cmap='gray')
        axs[2,0].scatter(pose_data_s2[i,0,:].cpu(), pose_data_s2[i,1,:].cpu(), s=0.07, c='red', alpha=0.6)
        axs[2,0].grid(False)
        axs[2,0].set_axis_off()

        axs[2,1].imshow(im_s2[i,:,:].cpu(), cmap='gray')
        axs[2,1].scatter(pose_recon_s2[i,0,:].cpu(), pose_recon_s2[i,1,:].cpu(), s=0.07, c='green', alpha=0.6)
        axs[2,1].grid(False)
        axs[2,1].set_axis_off()

        plt.savefig(image_dir + "/image_" + str(counter) + ".svg")
        plt.close()
        counter = counter + 1

def validate(model, dataloader, proj_params):
    model.eval()
    running_loss = 0.0
    counter = 0
    loss_array = []
    predicted_pose = [] 
    with torch.no_grad():
        #for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        for i, data in enumerate(dataloader):
            for p in range(0,1):
                im_three_channels, crop_coor_data, coor_3d_data = data
                im_three_channels = im_three_channels.to(device)
                #pose_data = pose_data.to(device)
                #eye_coor_data = eye_coor_data.to(device)
                coor_3d_data = coor_3d_data.to(device)
                proj_params = proj_params.to(device)
                coor_3d_data = coor_3d_data.to(device)
                crop_coor_data = crop_coor_data.to(device)
                crop_split = crop_coor_data[:,0,[0,2,4,6,8,10]]
                coor_3d = model(im_three_channels, crop_split)
                predicted_pose.append(coor_3d.cpu().numpy())
                pose_recon_b, pose_recon_s1, pose_recon_s2 = calc_proj_w_refra(coor_3d, proj_params)
                pose_data_b, pose_data_s1, pose_data_s2 = calc_proj_w_refra(coor_3d_data, proj_params)
                pose_recon_b = pose_recon_b.to(device)
                pose_recon_s1 = pose_recon_s1.to(device)
                pose_recon_s2 = pose_recon_s2.to(device)

                pose_data_b = pose_data_b.to(device)
                pose_data_s1 = pose_data_s1.to(device)
                pose_data_s2 = pose_data_s2.to(device)

                pose_recon_b[:,0,:] = pose_recon_b[:,0,:] - crop_coor_data[:,0,2,None]  # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_b[:,1,:] = pose_recon_b[:,1,:] - crop_coor_data[:,0,0,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s1[:,0,:] = pose_recon_s1[:,0,:] - crop_coor_data[:,0,6,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s1[:,1,:] = pose_recon_s1[:,1,:] - crop_coor_data[:,0,4,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s2[:,0,:] = pose_recon_s2[:,0,:] - crop_coor_data[:,0,10,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s2[:,1,:] = pose_recon_s2[:,1,:] - crop_coor_data[:,0,8,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices

                pose_data_b[:,0,:] = pose_data_b[:,0,:] - crop_coor_data[:,0,2,None]  # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_b[:,1,:] = pose_data_b[:,1,:] - crop_coor_data[:,0,0,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_s1[:,0,:] = pose_data_s1[:,0,:] - crop_coor_data[:,0,6,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_s1[:,1,:] = pose_data_s1[:,1,:] - crop_coor_data[:,0,4,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_s2[:,0,:] = pose_data_s2[:,0,:] - crop_coor_data[:,0,10,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_s2[:,1,:] = pose_data_s2[:,1,:] - crop_coor_data[:,0,8,None] # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_loss = criterion(coor_3d_data[:,:,0:10], coor_3d[:,:,0:10])
                eye_loss = torch.min(criterion(coor_3d_data[:,:,10:12], coor_3d[:,:,10:12]),criterion(coor_3d_data[:,:,10:12],torch.flip(coor_3d[:,:,10:12],(2,))))

                # Calculate loss in terms of distance error 
                pose_loss = torch.sum(pose_loss, dim=(1)); eye_loss = torch.sum(eye_loss, dim=(1))
                pose_loss = torch.sum(torch.sqrt(pose_loss), dim=(1)); eye_loss = torch.sum(torch.sqrt(eye_loss), dim=(1))
                loss = eye_loss + pose_loss
                loss_idx = (((loss.cpu().numpy()/12) > 0.3) & ((loss.cpu().numpy()/12) < 0.8)).nonzero()[0]
                dat_b = pose_data_b[loss_idx, :, :]
                dat_s1 = pose_data_s1[loss_idx, :, :]
                dat_s2 = pose_data_s2[loss_idx, :, :]
                recon_b = pose_recon_b[loss_idx, :, :]
                recon_s1 = pose_recon_s1[loss_idx, :, :]
                recon_s2 = pose_recon_s2[loss_idx, :, :]
                im_b = im_three_channels[loss_idx, 0, :, :].view(loss_idx.shape[0], imageSizeY, imageSizeY)
                im_s1 = im_three_channels[loss_idx, 1, :, :].view(loss_idx.shape[0], imageSizeY, imageSizeY)
                im_s2 = im_three_channels[loss_idx, 2, :, :].view(loss_idx.shape[0], imageSizeY, imageSizeY)
                if (np.random.rand(1) < 0):
                    display_fits(im_b, im_s1, im_s2, dat_b, dat_s1, dat_s2, recon_b, recon_s1, recon_s2, counter, output_dir)
                    counter = counter + loss_idx.shape[0]
                
                # Loss is an array of losses from all batches
                loss_array.append(loss.cpu().numpy())

                # Sum over all batches
                loss = torch.sum(loss)
                running_loss += loss.item()

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
                        axs[0,m].scatter(pose_data_b[m,0,:].cpu(), pose_data_b[m,1,:].cpu(), s=0.07, c='red', alpha=0.6)
             #          axs[0,m].scatter(eye_coor_data[m,0,0:2].cpu(), eye_coor_data[m,1,0:2].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[3,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                        axs[3,m].scatter(pose_recon_s1[m,0,:].cpu(), pose_recon_s1[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[2,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                        axs[2,m].scatter(pose_data_s1[m,0,:].cpu(), pose_data_s1[m,1,:].cpu(), s=0.07, c='red', alpha=0.6)
              #         axs[2,m].scatter(eye_coor_data[m,0,2:4].cpu(), eye_coor_data[m,1,2:4].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[5,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                        axs[5,m].scatter(pose_recon_s2[m,0,:].cpu(), pose_recon_s2[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                    for m in range(0,8):
                        axs[4,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                        axs[4,m].scatter(pose_data_s2[m,0,:].cpu(), pose_data_s2[m,1,:].cpu(), s=0.07, c='red', alpha=0.6)
               #        axs[4,m].scatter(eye_coor_data[m,0,4:6].cpu(), eye_coor_data[m,1,4:6].cpu(), s=0.07, c='green', alpha=0.6)
                    print('Saving',flush=True)
                    plt.savefig(output_dir + "/epoch_" + str(epoch) + ".svg")
                    plt.close()

    val_loss = running_loss/len(dataloader.dataset)
    return val_loss/12, loss_array, predicted_pose

train_loss = []
val_loss = []
predicted_pose_dict = {}
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    val_epoch_loss, loss_array, predicted_pose = validate(model, val_loader, proj_params)
    predicted_pose = np.concatenate(predicted_pose)
    loss_array = np.concatenate(loss_array, axis=0)
    print(val_epoch_loss)
    plt.hist((loss_array/12), bins=100, density=True, cumulative=True, histtype='step')
    #val_loss.append(val_epoch_loss)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)
    plt.xlabel('reconstruction error (mm)')
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1, 0.1))
    plt.savefig(output_dir + '/Histogram_cumulative.png')
    plt.close()
    print(loss_array.shape)
    plt.hist((loss_array/12), bins=100, density=True, cumulative=False)
    #val_loss.append(val_epoch_loss)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)
    plt.xlabel('reconstruction error (mm)')
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.xlim([0, 1])
    plt.xticks(np.arange(0, 1, 0.1))
    plt.savefig(output_dir + '/Histogram.png')
    predicted_pose_dict['predicted_pose'] = predicted_pose
    sio.savemat(output_dir + '/predicted_pose.mat',predicted_pose_dict)
    torch.save(predicted_pose, output_dir + '/predicted_pose.pt')
print('Exiting script')
