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
from CustomDataset_images_only import CustomImageDataset
from ResNet_Blocks_3D_five_blocks import resnet18
import time
from multiprocessing import Pool
import os
import scipy.io as sio
from scipy.optimize import least_squares
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=1, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="validations", type=str, help='path to store output images and plots')
parser.add_argument('-p','--proj_params', default="proj_params_101019_corrected_new", type=str, help='path to calibrated camera parameters')

imageSizeX = 141
imageSizeY = 141

date = '220726'

args = vars(parser.parse_args())

proj_params_path = args['proj_params']
epochs = args['epochs']
output_dir = args['output_dir']
proj_params = sio.loadmat(proj_params_path)
proj_params = torch.tensor(proj_params['proj_params'])
proj_params = proj_params[None,:,:]
proj_params_cpu = proj_params
lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 12, activation='leaky_relu').to(device)
n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 260*n_cuda

#if torch.cuda.device_count() > 1:
  #print("Using " + str(n_cuda) + " GPUs!")
#model = nn.DataParallel(model)

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

model.load_state_dict(torch.load('resnet_pose_220731_4.pt'))

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
im_folder = '../validation_data_3D_' + date + '_er_ob/images_real/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

coor_3d_folder = '../validation_data_3D_' + date + '_er_ob/annotations_' + date + '_coor_3d_tensor/'
coor_3d_files = sorted(os.listdir(coor_3d_folder))
coor_3d_files_add = [coor_3d_folder + file_name for file_name in coor_3d_files]

crop_coor_folder = '../validation_data_3D_' + date + '_er_ob/annotations_' + date + '_crop_coor_tensor/'
crop_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_files]


#data = CustomImageDataset(im_files_add, coor_3d_files_add, crop_coor_files_add, transform=transform)
#val_size = batch_size
#train_size = len(data) - val_size
#_, val_data = torch.utils.data.random_split(data, [train_size, val_size])

val_data = CustomImageDataset(im_files_add, coor_3d_files_add, crop_coor_files_add, transform=transform)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)

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

def triangulate_3d(pose_b, pose_s1, pose_s2, proj_params):
    coor = torch.zeros(2,3)
    coor[:,0] = pose_b
    coor[:,1] = pose_s1
    coor[:,2] = pose_s2

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
    
    fun = lambda x: ((fa1p00+fa1p10*x[2]+fa1p01*x[0]+fa1p20*x[2]**2+fa1p11*x[2]*x[0]+fa1p30*x[2]**3+fa1p21*x[2]**2*x[0] - coor[0,0])**2 +
            (fa2p00+fa2p10*x[2]+fa2p01*x[1]+fa2p20*x[2]**2+fa2p11*x[2]*x[1]+fa2p30*x[2]**3+fa2p21*x[2]**2*x[1] - coor[1,0])**2 +
            (fb1p00+fb1p10*x[0]+fb1p01*x[1]+fb1p20*x[0]**2+fb1p11*x[0]*x[1]+fb1p30*x[0]**3+fb1p21*x[0]**2*x[1] - coor[0,1])**2 +
            (fb2p00+fb2p10*x[0]+fb2p01*x[2]+fb2p20*x[0]**2+fb2p11*x[0]*x[2]+fb2p30*x[0]**3+fb2p21*x[0]**2*x[2] - coor[1,1])**2 +
            (fc1p00+fc1p10*x[1]+fc1p01*x[0]+fc1p20*x[1]**2+fc1p11*x[1]*x[0]+fc1p30*x[1]**3+fc1p21*x[1]**2*x[0] - coor[0,2])**2 +
            (fc2p00+fc2p10*x[1]+fc2p01*x[2]+fc2p20*x[1]**2+fc2p11*x[1]*x[2]+fc2p30*x[1]**3+fc2p21*x[1]**2*x[2] - coor[1,2])**2)
    bnds = ((-30,-30,50),(30,30,100))
    res = least_squares(fun, (0, 0, 70), method = 'dogbox', bounds = bnds)
    return torch.tensor(res.x), torch.tensor(res.fun)

def save_images(pose_recon_b, pose_recon_s1, pose_recon_s2, pose_data_b, pose_data_s1, pose_data_s2, im_b, im_s1, im_s2, counter, output_dir):
    for i in range(pose_recon_b.shape[0]):
        _,axs = plt.subplots(nrows=3, ncols=2)
        axs[0,1].imshow(im_b[i,:,:].cpu(), cmap='gray')
        axs[0,1].scatter(pose_recon_b[i,0,:], pose_recon_b[i,1,:], s=0.07, c='green', alpha=0.6)
        axs[0,1].grid(False)
        axs[0,1].set_axis_off()
        axs[0,0].imshow(im_b[i,:,:].cpu(), cmap='gray')
        axs[0,0].scatter(pose_data_b[i,0,:], pose_data_b[i,1,:], s=0.07, c='red', alpha=0.6)
        axs[0,0].grid(False)
        axs[0,0].set_axis_off()

        axs[1,1].imshow(im_s1[i,:,:].cpu(), cmap='gray')
        axs[1,1].scatter(pose_recon_s1[i,0,:], pose_recon_s1[i,1,:], s=0.07, c='green', alpha=0.6)
        axs[1,1].grid(False)
        axs[1,1].set_axis_off()
        axs[1,0].imshow(im_s1[i,:,:].cpu(), cmap='gray')
        axs[1,0].scatter(pose_data_s1[i,0,:], pose_data_s1[i,1,:], s=0.07, c='red', alpha=0.6)
        axs[1,0].grid(False)
        axs[1,0].set_axis_off()

        axs[2,1].imshow(im_s2[i,:,:].cpu(), cmap='gray')
        axs[2,1].scatter(pose_recon_s2[i,0,:], pose_recon_s2[i,1,:], s=0.07, c='green', alpha=0.6)
        axs[2,1].grid(False)
        axs[2,1].set_axis_off()
        axs[2,0].imshow(im_s2[i,:,:].cpu(), cmap='gray')
        axs[2,0].scatter(pose_data_s2[i,0,:], pose_data_s2[i,1,:], s=0.07, c='red', alpha=0.6)
        axs[2,0].grid(False)
        axs[2,0].set_axis_off()

        plt.savefig(output_dir + '/im_' + str(counter) + '.svg')
        plt.close()
        counter = counter + 1




def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def validate(model, dataloader, proj_params):
    model.eval()
    running_loss = 0.0
    pose_loss_array = []
    counter = 0
    with torch.no_grad():
        #for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        for i, data in enumerate(dataloader):
            for p in range(0,1): 
                print(i,flush=True)
                im_three_channels_cpu,coor_3d_data_cpu,crop_coor_data_cpu = data
                im_three_channels = im_three_channels_cpu.to(device)
                coor_3d_data = coor_3d_data_cpu.to(device)
                crop_coor_data = crop_coor_data_cpu.to(device)
                crop_split = crop_coor_data[:,0,[0,2,4,6,8,10]]
                proj_params = proj_params_cpu.to(device)
                pose_recon_b, pose_recon_s1, pose_recon_s2 = model(im_three_channels)
                pose_data_b, pose_data_s1, pose_data_s2 = calc_proj_w_refra(coor_3d_data,proj_params)
                
                pose_recon_b = pose_recon_b.to(device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s1 = pose_recon_s1.to(device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_recon_s2 = pose_recon_s2.to(device) # This is because calc_proj_w_refra is calibrated for MATLAB indices
                pose_data_b = pose_data_b.to(device)
                pose_data_s1 = pose_data_s1.to(device)
                pose_data_s2 = pose_data_s2.to(device)
                pose_data_b[:,0,:] = pose_data_b[:,0,:] - crop_coor_data[:,0,2,None] 
                pose_data_b[:,1,:] = pose_data_b[:,1,:] - crop_coor_data[:,0,0,None] 
                pose_data_s1[:,0,:] = pose_data_s1[:,0,:] - crop_coor_data[:,0,6,None] 
                pose_data_s1[:,1,:] = pose_data_s1[:,1,:] - crop_coor_data[:,0,4,None] 
                pose_data_s2[:,0,:] = pose_data_s2[:,0,:] - crop_coor_data[:,0,10,None] 
                pose_data_s2[:,1,:] = pose_data_s2[:,1,:] - crop_coor_data[:,0,8,None] 
                #eye_loss_b = torch.min(criterion(pose_recon_b[:,:,10:12],pose_data_b[:,:,10:12]), criterion(pose_recon_b[:,:,10:12], torch.flip(pose_data_b[:,:,0:2],(2,))))
                #eye_loss_s1 = torch.min(criterion(pose_recon_s1[:,:,10:12],pose_data_s1[:,:,10:12]), criterion(pose_recon_s1[:,:,10:12], torch.flip(pose_data_s1[:,:,10:12],(2,))))
                #eye_loss_s2 = torch.min(criterion(pose_recon_s2[:,:,10:12],pose_data_s2[:,:,10:12]), criterion(pose_recon_s2[:,:,10:12], torch.flip(pose_data_s2[:,:,10:12],(2,))))
                #eye_loss = eye_loss_b + eye_loss_s1 + eye_loss_s2
                
                pose_recon_b[:,0,:] = pose_recon_b[:,0,:] + crop_coor_data[:,0,2,None]
                pose_recon_b[:,1,:] = pose_recon_b[:,1,:] + crop_coor_data[:,0,0,None]
                pose_recon_s1[:,0,:] = pose_recon_s1[:,0,:] + crop_coor_data[:,0,6,None]
                pose_recon_s1[:,1,:] = pose_recon_s1[:,1,:] + crop_coor_data[:,0,4,None]
                pose_recon_s2[:,0,:] = pose_recon_s2[:,0,:] + crop_coor_data[:,0,10,None]
                pose_recon_s2[:,1,:] = pose_recon_s2[:,1,:] + crop_coor_data[:,0,8,None]
                pose_recon_b = pose_recon_b.cpu(); pose_recon_s1 = pose_recon_s1.cpu(); pose_recon_s2 = pose_recon_s2.cpu()
                pose_rerecon_b = torch.zeros(pose_recon_b.shape); pose_rerecon_s1 = torch.zeros(pose_recon_s1.shape); pose_rerecon_s2 = torch.zeros(pose_recon_s2.shape)
                
                rerecon_coor_3d = torch.zeros(coor_3d_data_cpu.shape)
                #proj_params_cpu = torch.tensor(proj_params_cpu)
                begin_time = time.time()
                for batch_sample in range(0, pose_recon_b.shape[0]):
                    for point in range(0,10):
                        rerecon_coor_3d[batch_sample, :, point], _ = triangulate_3d(pose_recon_b[batch_sample, :, point] + 1, pose_recon_s1[batch_sample, :, point] + 1, pose_recon_s2[batch_sample, :, point] + 1, proj_params_cpu)
                    for point in range(10, 12):
                        min_cost = torch.tensor(float('inf'))
                        for perm_idx1 in range(10,12):
                            for perm_idx2 in range(10,12):
                                recon_coor_3d_temp, cost = triangulate_3d(pose_recon_b[batch_sample, :, point] + 1, pose_recon_s1[batch_sample, :, perm_idx1] + 1, pose_recon_s2[batch_sample, :, perm_idx2] + 1, proj_params_cpu)
                                if (cost < min_cost):
                                    rerecon_coor_3d[batch_sample, :, point] = recon_coor_3d_temp
                                    min_cost = cost
                print(time.time() - begin_time)
                pose_rerecon_b, pose_rerecon_s1, pose_rerecon_s2 = calc_proj_w_refra(rerecon_coor_3d, proj_params_cpu)
                pose_rerecon_b[:,0,:] = pose_rerecon_b[:,0,:] - crop_coor_data_cpu[:,0,2,None]
                pose_rerecon_b[:,1,:] = pose_rerecon_b[:,1,:] - crop_coor_data_cpu[:,0,0,None]
                pose_rerecon_s1[:,0,:] = pose_rerecon_s1[:,0,:] - crop_coor_data_cpu[:,0,6,None]
                pose_rerecon_s1[:,1,:] = pose_rerecon_s1[:,1,:] - crop_coor_data_cpu[:,0,4,None]
                pose_rerecon_s2[:,0,:] = pose_rerecon_s2[:,0,:] - crop_coor_data_cpu[:,0,10,None]
                pose_rerecon_s2[:,1,:] = pose_rerecon_s2[:,1,:] - crop_coor_data_cpu[:,0,8,None]

                pose_recon_b = pose_rerecon_b; pose_recon_s1 = pose_rerecon_s1; pose_recon_s2 = pose_rerecon_s2

                pose_loss = criterion(rerecon_coor_3d, coor_3d_data_cpu)
                pose_loss = torch.sqrt(torch.sum(pose_loss, dim=(1)))
                pose_loss = torch.sum(pose_loss, dim=(1))/12
                loss_idx = (pose_loss > 0.35) & (pose_loss < 0.5)
                im_b = im_three_channels[loss_idx,0,:,:]; im_s1 = im_three_channels[loss_idx,1,:,:]; im_s2 = im_three_channels[loss_idx,2,:,:]; 
                save_images(pose_recon_b[loss_idx,:,:], pose_recon_s1[loss_idx,:,:], pose_recon_s2[loss_idx,:,:], pose_data_b[loss_idx,:,:].cpu(), pose_data_s1[loss_idx,:,:].cpu(), pose_data_s2[loss_idx,:,:].cpu(), im_b, im_s1, im_s2, counter, output_dir)
                counter = counter + im_b.shape[0]
                #pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data_b[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data_s1[:,:,0:10]) + criterion(pose_recon_s2[:,:,0:10], pose_data_s2[:,:,0:10])
                #eye_loss_b = criterion(pose_recon_b[:,:,10:12],pose_data_b[:,:,10:12])
                #eye_loss_s1 = criterion(pose_recon_s1[:,:,10:12],pose_data_s1[:,:,10:12])
                #eye_loss_s2 = criterion(pose_recon_s2[:,:,10:12],pose_data_s2[:,:,10:12])
                #eye_loss = eye_loss_b + eye_loss_s1 + eye_loss_s2
                #pose_loss = torch.sum(pose_loss, dim=(1,2)); eye_loss = torch.sum(eye_loss, dim=(1,2))
                #loss = pose_loss + eye_loss
                #loss = torch.sqrt(loss)/12


                #pose_loss = torch.sum(pose_loss, dim=(1,2)); eye_loss = torch.sum(eye_loss, dim=(1,2))
                #loss = pose_loss + eye_loss
                #max_idx = torch.argmax(loss).cpu()
                #print(max_idx)
                #print(torch.max(loss))
                loss = torch.sum(torch.sum(pose_loss))
                pose_loss_array.append(pose_loss)
                running_loss += loss.item()
                #running_pose_loss += pose_loss.item()
                
                # save the last batch input and output of every epoch
                if i == int(len(val_data)/dataloader.batch_size) - 1:
                    num_rows = 8
                    im_b = im_three_channels[:,0,:,:]
                    im_s1 = im_three_channels[:,1,:,:]
                    im_s2 = im_three_channels[:,2,:,:]
                    images_b = im_b.view(batch_size,1,imageSizeY,imageSizeX).cpu()
                    images_s1 = im_s1.view(batch_size,1,imageSizeY,imageSizeX).cpu()
                    images_s2 = im_s2.view(batch_size,1,imageSizeY,imageSizeX).cpu()
                    _,axs = plt.subplots(nrows=6, ncols=8)
                    #perm = torch.randperm(images_b.size(0))
                    #idx = perm[:8]#*0 + max_idx
                    idx = torch.arange(0,8)

                    # Overlay pose
                    for m in range(0,8):
                        axs[1,m].imshow(images_b[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[1,m].scatter(pose_recon_b[idx[m],0,:].cpu(), pose_recon_b[idx[m],1,:].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[1,m].grid(False)
                        axs[1,m].set_axis_off()

                    for m in range(0,8):
                        axs[0,m].imshow(images_b[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[0,m].scatter(pose_data_b[idx[m],0,0:12].cpu(), pose_data_b[idx[m],1,0:12].cpu(), s=0.07, c='red', alpha=0.6)
             #          axs[0,m].scatter(eye_coor_data[m,0,0:2].cpu(), eye_coor_data[m,1,0:2].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[0,m].grid(False)
                        axs[0,m].set_axis_off()
                    for m in range(0,8):
                        axs[3,m].imshow(images_s1[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[3,m].scatter(pose_recon_s1[idx[m],0,:].cpu(), pose_recon_s1[idx[m],1,:].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[3,m].grid(False)
                        axs[3,m].set_axis_off()

                    for m in range(0,8):
                        axs[2,m].imshow(images_s1[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[2,m].scatter(pose_data_s1[idx[m],0,0:12].cpu(), pose_data_s1[idx[m],1,0:12].cpu(), s=0.07, c='red', alpha=0.6)
              #         axs[2,m].scatter(eye_coor_data[m,0,2:4].cpu(), eye_coor_data[m,1,2:4].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[2,m].grid(False)
                        axs[2,m].set_axis_off()

                    for m in range(0,8):
                        axs[5,m].imshow(images_s2[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[5,m].scatter(pose_recon_s2[idx[m],0,:].cpu(), pose_recon_s2[idx[m],1,:].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[5,m].grid(False)
                        axs[5,m].set_axis_off()

                    for m in range(0,8):
                        axs[4,m].imshow(images_s2[idx[m],0,:,:].cpu(), cmap='gray')
                        axs[4,m].scatter(pose_data_s2[idx[m],0,0:12].cpu(), pose_data_s2[idx[m],1,0:12].cpu(), s=0.07, c='red', alpha=0.6)
               #        axs[4,m].scatter(eye_coor_data[m,0,4:6].cpu(), eye_coor_data[m,1,4:6].cpu(), s=0.07, c='green', alpha=0.6)
                        axs[4,m].grid(False)
                        axs[4,m].set_axis_off()
                    plt.savefig(output_dir + "/epoch_" + str(idx[0]) + ".svg")
                    plt.close()

    val_loss = running_loss/len(dataloader.dataset)
    return val_loss, pose_loss_array

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    val_epoch_loss, pose_loss_array = validate(model, val_loader, proj_params)
    pose_loss_array = np.concatenate(pose_loss_array)
    plt.hist(pose_loss_array, bins='auto', density=True)
    plt.savefig(output_dir + '/histogram.svg')
    plt.ylim([0, 50])
    plt.xlim([0, 2])
    plt.close()
    #val_loss.append(val_epoch_loss)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)
print('Results saved, exiting script')

