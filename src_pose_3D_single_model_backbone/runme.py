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
parser.add_argument('-e','--epochs',default=2, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="outputs/220705_symmetric_eyes", type=str, help='path to store output images and plots')
parser.add_argument('-p','--proj_params', default="proj_params_101019_corrected_new", type=str, help='path to calibrated camera parameters')


date = '220717'
args = vars(parser.parse_args())
imageSizeX = 141
imageSizeY = 141

epochs = args['epochs']
output_dir = args['output_dir']
proj_params_path = args['proj_params']

proj_params = sio.loadmat(proj_params_path)
proj_params = proj_params['proj_params']
lr = 0.001

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)
    print('Creating new directory to store output images')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 12, activation='leaky_relu').to(device)

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 350*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
pose_folder = '../lookup_table_head/training_data_3D_220717/annotations_'+date+'_pose_tensor/'
pose_files = sorted(os.listdir(pose_folder))
pose_files_add = [pose_folder + file_name for file_name in pose_files]

crop_coor_folder = '../lookup_table_head/training_data_3D_220717/annotations_'+date+'_crop_coor_tensor/'
crop_coor_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_coor_files]

eye_coor_folder = '../lookup_table_head/training_data_3D_220717/annotations_'+date+'_eye_coor_tensor/'
eye_coor_files = sorted(os.listdir(eye_coor_folder))
eye_coor_files_add = [eye_coor_folder + file_name for file_name in eye_coor_files]

coor_3d_folder = '../lookup_table_head/training_data_3D_220717/annotations_'+date+'_coor_3d_tensor/'
coor_3d_files = sorted(os.listdir(coor_3d_folder))
coor_3d_files_add = [coor_3d_folder + file_name for file_name in coor_3d_files]


im_folder = '../lookup_table_head/training_data_3D_220717/images/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

data = CustomImageDataset(im_files_add, pose_files_add, eye_coor_files_add, crop_coor_files_add, coor_3d_files_add, transform=transform)
train_size = int(len(data)*0.9)
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)
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
    return coor_b - 1, coor_s1 - 1, coor_s2 - 1

def final_loss(mse_loss, mu, logvar):
    MSE = mse_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def fit(model, dataloader, proj_params):
    model.train()
    running_loss = 0.0
    running_pose_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        im_three_channels, pose_data, eye_coor_data, crop_coor_data, coor_3d_data = data
        crop_split = crop_coor_data[:,0,[0,2,4,6,8,10]]
        im_three_channels = im_three_channels.to(device)
        pose_data = pose_data.to(device)
        eye_coor_data = eye_coor_data.to(device)
        crop_split = crop_split.to(device)
        proj_params = proj_params.to(device)
        coor_3d_data = coor_3d_data.to(device)
        optimizer.zero_grad()
        coor_3d = model(im_three_channels, crop_split)
        pose_recon_b, pose_recon_s1, pose_recon_s2 = calc_proj_w_refra(coor_3d, proj_params)
        pose_recon_b = pose_recon_b.to(pose_data.device)
        pose_recon_s1 = pose_recon_s1.to(pose_data.device)
        pose_recon_s2 = pose_recon_s2.to(pose_data.device)
        #pose_recon_b[:,0,:] = pose_recon_b[:,0,:] - crop_split[:,1,None] 
        #pose_recon_b[:,1,:] = pose_recon_b[:,1,:] - crop_split[:,0,None] 
        #pose_recon_s1[:,0,:] = pose_recon_s1[:,0,:] - crop_split[:,3,None] 
        #pose_recon_s1[:,1,:] = pose_recon_s1[:,1,:] - crop_split[:,2,None] 
        #pose_recon_s2[:,0,:] = pose_recon_s2[:,0,:] - crop_split[:,5,None] 
        #pose_recon_s2[:,1,:] = pose_recon_s2[:,1,:] - crop_split[:,4,None]
        #pose_data_b = pose_data_b.to(device)
        #pose_data_s1 = pose_data_s1.to(device)
        #pose_data_s2 = pose_data_s2.to(device)
        #pose_data_b[:,0,:] = pose_data_b[:,0,:] - crop_split[:,0,1,None]
        #pose_data_b[:,1,:] = pose_data_b[:,1,:] - crop_split[:,0,0,None]
        #pose_data_s1[:,0,:] = pose_data_s1[:,0,:] - crop_split[:,0,3,None]
        #pose_data_s1[:,1,:] = pose_data_s1[:,1,:] - crop_split[:,0,2,None]
        #pose_data_s2[:,0,:] = pose_data_s2[:,0,:] - crop_split[:,0,5,None]
        #pose_data_s2[:,1,:] = pose_data_s2[:,1,:] - crop_split[:,0,4,None]
        #pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,20:30])
        #eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
        pose_loss = criterion(coor_3d_data[:,:,0:10], coor_3d[:,:,0:10])
        eye_loss = torch.min(criterion(coor_3d_data[:,:,10:12], coor_3d[:,:,10:12]),criterion(coor_3d_data[:,:,10:12],torch.flip(coor_3d[:,:,10:12],(2,))))
        loss = pose_loss + eye_loss
        running_loss += loss.item()
        running_pose_loss += pose_loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    pose_loss = running_pose_loss/len(dataloader.dataset)
    return train_loss, pose_loss


def validate(model, dataloader, proj_params):
    model.eval()
    running_loss = 0.0
    running_pose_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im_three_channels, pose_data, eye_coor_data, crop_coor_data, coor_3d_data = data
            im_three_channels = im_three_channels.to(device)
            pose_data = pose_data.to(device)
            eye_coor_data = eye_coor_data.to(device)
            coor_3d_data = coor_3d_data.to(device)
            proj_params = proj_params.to(device)
            coor_3d_data = coor_3d_data.to(device)
            crop_coor_data = crop_coor_data.to(device)
            crop_split = crop_coor_data[:,0,[0,2,4,6,8,10]]
            coor_3d = model(im_three_channels, crop_split)
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


            #pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,20:30])
            #eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
            pose_loss = criterion(coor_3d_data[:,:,0:10], coor_3d[:,:,0:10])
            eye_loss = torch.min(criterion(coor_3d_data[:,:,10:12], coor_3d[:,:,10:12]),criterion(coor_3d_data[:,:,10:12],torch.flip(coor_3d[:,:,10:12],(2,))))

            loss = pose_loss + eye_loss
            running_loss += loss.item()
            running_pose_loss += pose_loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                #both = torch.cat((pose_data.view(batch_size, 1, 2, 10)[:8],
                #                  pose_reconstruction.view(batch_size, 1, 2, 10)[:8]))
                #torch.save(both, output_dir + "/pose_" + str(epoch) + ".png")
                im_b = im_three_channels[:,0,:,:]
                im_s1 = im_three_channels[:,1,:,:]
                im_s2 = im_three_channels[:,2,:,:]
                images_b = im_b.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                images_s1 = im_s1.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                images_s2 = im_s2.view(batch_size,1,imageSizeY,imageSizeX)[:8]
                _, axs = plt.subplots(nrows=6, ncols=8)

                # Overlay pose
                for m in range(0,8):
                    axs[1,m].imshow(images_b[m,0,:,:].cpu(), cmap='gray')
                    axs[1,m].scatter(pose_recon_b[m,0,:].cpu(), pose_recon_b[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[1,m].axis('off')

                for m in range(0,8):
                    axs[0,m].imshow(images_b[m,0,:,:].cpu(), cmap='gray')
                    axs[0,m].scatter(pose_data_b[m,0,0:10].cpu(), pose_data_b[m,1,0:10].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[0,m].scatter(pose_data_b[m,0,10:12].cpu(), pose_data_b[m,1,10:12].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[0,m].axis('off')

                for m in range(0,8):
                    axs[3,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                    axs[3,m].scatter(pose_recon_s1[m,0,:].cpu(), pose_recon_s1[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[3,m].axis('off')

                for m in range(0,8):
                    axs[2,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                    axs[2,m].scatter(pose_data_s1[m,0,0:10].cpu(), pose_data_s1[m,1,0:10].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[2,m].scatter(pose_data_s1[m,0,10:12].cpu(), pose_data_s1[m,1,10:12].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[2,m].axis('off')
                
                for m in range(0,8):
                    axs[5,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                    axs[5,m].scatter(pose_recon_s2[m,0,:].cpu(), pose_recon_s2[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[5,m].axis('off')

                for m in range(0,8):
                    axs[4,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                    axs[4,m].scatter(pose_data_s2[m,0,0:10].cpu(), pose_data_s2[m,1,0:10].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[4,m].scatter(pose_data_s2[m,0,10:12].cpu(), pose_data_s2[m,1,10:12].cpu(), s=0.07, c='green', alpha=0.6)
                    axs[4,m].axis('off')


                plt.axis('off')
                plt.savefig(output_dir + "/epoch_" + str(epoch) + ".svg")
                plt.close()
                #save_image(images.cpu(), output_dir + "/output_" + str(epoch) + ".png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    pose_loss = running_pose_loss/len(dataloader.dataset)
    return val_loss, pose_loss

train_loss = []
val_loss = []
train_pose_loss_array = []
val_pose_loss_array = []

proj_params = torch.tensor(proj_params)
proj_params = proj_params[None, :]

lowest_val_loss = 5

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    train_epoch_loss, train_pose_loss = fit(model, train_loader, proj_params)
    val_epoch_loss, val_pose_loss = validate(model, val_loader, proj_params)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_pose_loss_array.append(train_pose_loss)
    val_pose_loss_array.append(val_pose_loss)
    if (val_epoch_loss < lowest_val_loss):
        torch.save(model.state_dict(), 'resnet_pose_' + date + '_2_lowest_loss.pt')
        print('Saving model with loss = ' + str(val_epoch_loss))
        lowest_val_loss = val_epoch_loss
    print(f"Train Loss: {train_epoch_loss:.4f}",flush=True)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)

torch.save(model.state_dict(), 'resnet_pose_' + date + '_2.pt')

plt.plot(train_loss[2:], color='green')
plt.plot(val_loss[2:], color='red')
plt.plot(train_pose_loss_array[10:], linestyle='--', color='green')
plt.plot(val_pose_loss_array[10:], linestyle='--', color='red')
plt.savefig(output_dir + "/loss_truncated.png")

plt.plot(train_loss, color='green')
plt.plot(val_loss, color='red')
plt.savefig(output_dir + "/loss.png")

