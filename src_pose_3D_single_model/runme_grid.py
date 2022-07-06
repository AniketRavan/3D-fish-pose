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
from ResNet_Blocks_3D_no_constraints import resnet18
import time
from multiprocessing import Pool
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=10, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="outputs/220704", type=str, help='path to store output images and plots')

args = vars(parser.parse_args())
imageSizeX = 141
imageSizeY = 141

epochs = args['epochs']
output_dir = args['output_dir']

lr = 0.01
date = '220629'

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)
    print('Creating new directory to store output images')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(5, 12, activation='leaky_relu').to(device)

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 390*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])
pose_folder = '../lookup_table_head/training_data_3D_pose_shifted_1/annotations_' + date + '_pose_tensor/'
pose_files = sorted(os.listdir(pose_folder))
pose_files_add = [pose_folder + file_name for file_name in pose_files]

crop_coor_folder = '../lookup_table_head/training_data_3D_pose_shifted_1/annotations_' + date + '_pose_tensor/'
crop_coor_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_coor_files]

eye_coor_folder = '../lookup_table_head/training_data_3D_pose_shifted_1/annotations_' + date + '_eye_coor_tensor/'
eye_coor_files = sorted(os.listdir(eye_coor_folder))
eye_coor_files_add = [eye_coor_folder + file_name for file_name in eye_coor_files]

im_folder = '../lookup_table_head/training_data_3D_pose_shifted_1/images/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

x = torch.arange(0,imageSizeX)
y = torch.arange(0,imageSizeY)
x_grid, y_grid = torch.meshgrid(x,y)
im_grid = torch.zeros(2,imageSizeY,imageSizeX)
im_grid[0,:,:] = x_grid
im_grid[1,:,:] = y_grid

data = CustomImageDataset(im_files_add, pose_files_add, eye_coor_files_add, crop_coor_files_add, transform=transform)
train_size = int(len(data)*0.9)
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=n_cuda*12,prefetch_factor=2,persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=n_cuda*12,prefetch_factor=2,persistent_workers=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')


def final_loss(mse_loss, mu, logvar):
    MSE = mse_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    running_pose_loss = 0.0
    #for i, data in enumerate(dataloader):
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        im_three_channels, pose_data, eye_coor_data, crop_coor_data = data
        im_grid_3d = torch.zeros(im_three_channels.shape[0], 2, im_three_channels.shape[2], im_three_channels.shape[3])
        im_grid_3d[:,0,:,:] = im_grid[0,:,:]
        im_grid_3d[:,1,:,:] = im_grid[1,:,:]
        im_five_channels = torch.cat((im_three_channels, im_grid_3d), 1)
        im_five_channels = im_five_channels.to(device)
        eye_coor_data = eye_coor_data.to(device)
        crop_coor_data = crop_coor_data.to(device)
        optimizer.zero_grad()
        pose_recon_b, pose_recon_s1, pose_recon_s2 = model(im_five_channels)
        pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,20:30])
        eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
        loss = pose_loss + 5*eye_loss
        running_loss += loss.item()
        running_pose_loss += pose_loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    train_pose_loss = running_pose_loss/len(dataloader.dataset)
    return train_loss, train_pose_loss


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    running_pose_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im_three_channels, pose_data, eye_coor_data, crop_coor_data = data
            im_grid_3d = torch.zeros(im_three_channels.shape[0], 2, im_three_channels.shape[2], im_three_channels.shape[3])
            im_grid_3d[:,0,:,:] = im_grid[0,:,:]
            im_grid_3d[:,1,:,:] = im_grid[1,:,:]
            im_five_channels = torch.cat((im_three_channels, im_grid_3d), 1)
            im_five_channels = im_five_channels.to(device)
            pose_data = pose_data.to(device)
            eye_coor_data = eye_coor_data.to(device)
            crop_coor_data = crop_coor_data.to(device)
            pose_recon_b, pose_recon_s1, pose_recon_s2 = model(im_five_channels)
            pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,20:30])
            eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
            loss = pose_loss + 5*eye_loss
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

                for m in range(0,8):
                    axs[0,m].imshow(images_b[m,0,:,:].cpu(), cmap='gray')
                    axs[0,m].scatter(pose_data[m,0,0:10].cpu(), pose_data[m,1,0:10].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[0,m].scatter(eye_coor_data[m,0,0:2].cpu(), eye_coor_data[m,1,0:2].cpu(), s=0.07, c='green', alpha=0.6)

                for m in range(0,8):
                    axs[3,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                    axs[3,m].scatter(pose_recon_s1[m,0,:].cpu(), pose_recon_s1[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                for m in range(0,8):
                    axs[2,m].imshow(images_s1[m,0,:,:].cpu(), cmap='gray')
                    axs[2,m].scatter(pose_data[m,0,10:20].cpu(), pose_data[m,1,10:20].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[2,m].scatter(eye_coor_data[m,0,2:4].cpu(), eye_coor_data[m,1,2:4].cpu(), s=0.07, c='green', alpha=0.6)
                
                for m in range(0,8):
                    axs[5,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                    axs[5,m].scatter(pose_recon_s2[m,0,:].cpu(), pose_recon_s2[m,1,:].cpu(), s=0.07, c='green', alpha=0.6)

                for m in range(0,8):
                    axs[4,m].imshow(images_s2[m,0,:,:].cpu(), cmap='gray')
                    axs[4,m].scatter(pose_data[m,0,20:30].cpu(), pose_data[m,1,20:30].cpu(), s=0.07, c='red', alpha=0.6)
                    axs[4,m].scatter(eye_coor_data[m,0,4:6].cpu(), eye_coor_data[m,1,4:6].cpu(), s=0.07, c='green', alpha=0.6)



                plt.savefig(output_dir + "/epoch_" + str(epoch) + ".svg")
                plt.close()
                #save_image(images.cpu(), output_dir + "/output_" + str(epoch) + ".png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    val_pose_loss = running_pose_loss/len(dataloader.dataset)
    return val_loss, val_pose_loss

train_loss = []
val_loss = []
train_pose_loss_array = []
val_pose_loss_array = []

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    train_epoch_loss, train_pose_loss = fit(model, train_loader)
    val_epoch_loss, val_pose_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_pose_loss_array.append(train_pose_loss)
    val_pose_loss_array.append(val_pose_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}",flush=True)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)

torch.save(model.state_dict(), 'resnet_pose_' + date + '_grid.pt')
print(type(train_pose_loss_array))

plt.plot(train_loss[20:], color='green')
plt.plot(val_loss[20:], color='red')
plt.plot(train_pose_loss_array[20:], linestyle='--', color='green')
plt.plot(val_pose_loss_array[20:], linestyle='--', color='red')
plt.savefig(output_dir + "/loss_truncated.png")

plt.plot(train_loss, color='green')
plt.plot(val_loss, color='red')
plt.savefig(output_dir + "/loss.png")

