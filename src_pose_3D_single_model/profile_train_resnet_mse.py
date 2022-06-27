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
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=10, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-o','--output_dir', default="outputs/pose_outputs", type=str, help='path to store output images and plots')

args = vars(parser.parse_args())
imageSizeX = 141
imageSizeY = 141

epochs = args['epochs']
output_dir = args['output_dir']

lr = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(1, 12, activation='leaky_relu').to(device)

n_cuda = torch.cuda.device_count()
if (torch.cuda.is_available()):
    print(str(n_cuda) + 'GPUs are available!')
else: print('Cuda is not available')
batch_size = 175*n_cuda

if torch.cuda.device_count() > 1:
  print("Using " + str(n_cuda) + " GPUs!")
  model = nn.DataParallel(model)

transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float)])

pose_folder = '../lookup_table_head/training_data_3D_pose/annotations_220603_pose_tensor/'
pose_files = sorted(os.listdir(pose_folder))
pose_files_add = [pose_folder + file_name for file_name in pose_files]

crop_coor_folder = '../lookup_table_head/training_data_3D_pose/annotations_220603_pose_tensor/'
crop_coor_files = sorted(os.listdir(crop_coor_folder))
crop_coor_files_add = [crop_coor_folder + file_name for file_name in crop_coor_files]

eye_coor_folder = '../lookup_table_head/training_data_3D_pose/annotations_220603_eye_coor_tensor/'
eye_coor_files = sorted(os.listdir(eye_coor_folder))
eye_coor_files_add = [eye_coor_folder + file_name for file_name in eye_coor_files]

im_folder = '../lookup_table_head/training_data_3D_pose/images/'
im_files = sorted(os.listdir(im_folder))
im_files_add = [im_folder + file_name for file_name in im_files]

data = CustomImageDataset(im_files_add, pose_files_add, eye_coor_files_add, crop_coor_files_add, transform=transform)
train_size = int(len(data)*0.9)
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)
val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=False,num_workers=n_cuda*16,prefetch_factor=2,persistent_workers=True)


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='sum')

def final_loss(mse_loss, mu, logvar):
    MSE = mse_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    #for i, data in enumerate(dataloader):
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        im_b, im_s1, im_s2, pose_data, eye_coor_data, crop_coor_data = data
        #im_b = image_data[:,:,:141,:]
        #im_s1 = image_data[:,:,141:282,:]
        #im_s2 = image_data[:,:,282:,:]
        im_b = im_b.to(device)
        im_s1 = im_s1.to(device)
        im_s2 = im_s2.to(device)
        pose_data = pose_data.to(device)
        eye_coor_data = eye_coor_data.to(device)
        crop_coor_data = crop_coor_data.to(device)
        optimizer.zero_grad()
        pose_recon_b, pose_recon_s1, pose_recon_s2 = model(im_b,im_s1,im_s2)
        pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,21:31])
        eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
        loss = pose_loss + eye_loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            im_b, im_s1, im_s2, pose_data, eye_coor_data, crop_coor_data = data
            im_b = im_b.to(device)
            im_s1 = im_s1.to(device)
            im_s2 = im_s2.to(device)
            pose_data = pose_data.to(device)
            eye_coor_data = eye_coor_data.to(device)
            crop_coor_data = crop_coor_data.to(device)
            pose_recon_b, pose_recon_s1, pose_recon_s2 = model(im_b,im_s1,im_s2)
            pose_loss = criterion(pose_recon_b[:,:,0:10], pose_data[:,:,0:10]) + criterion(pose_recon_s1[:,:,0:10], pose_data[:,:,10:20]) + criterion(pose_recon_s2[:,:,0:10], pose_data[:,:,21:31])
            eye_loss = criterion(pose_recon_b[:,:,10:12],eye_coor_data[:,:,0:2]) + criterion(pose_recon_s1[:,:,10:12],eye_coor_data[:,:,2:4]) + criterion(pose_recon_s2[:,:,10:12],eye_coor_data[:,:,4:6])
            loss = pose_loss + eye_loss
            running_loss += loss.item()

    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

train_loss = []
val_loss = []

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("traces/trace_" + str(p.step_num) + ".json")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=torch.profiler.schedule(wait=1,warmup=1,active=1), record_shapes=False,on_trace_ready=trace_handler) as prof:
    for epoch in range(0,3):
        print(f"Epoch {epoch+1} of 8")
        train_epoch_loss = fit(model, train_loader)
        train_loss.append(train_epoch_loss)
        prof.step()
    print(f"Train Loss: {train_epoch_loss:.4f}",flush=True)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

