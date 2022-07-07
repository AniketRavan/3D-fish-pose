import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import conv_model
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from CustomDataset import CustomImageDataset
from ResNet_Blocks import resnet18
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs',default=10, type=int, help='number of epochs to train the VAE for')
parser.add_argument('-f','--tail_error_factor', default=1.0, type=float, help='factor to scale the tail error by')
parser.add_argument('-o','--output_dir', default="../outputs_resnet", type=str, help='path to store output images and plots')

args = vars(parser.parse_args())
tail_error_factor = args['tail_error_factor']
output_dir = args['output_dir']

if (not os.path.isdir(output_dir)):
    os.mkdir(output_dir)

epochs = args['epochs']
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = conv_model.VAE().to(device)
if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
batch_size = 1000*torch.cuda.device_count()
if (torch.cuda.is_available()):
    print('Cuda is available!')
else: print('Cuda is not available')
transform = transforms.Compose([transforms.ToTensor()])

#train_data = datasets.MNIST(root='../input/data',train=True,download=True,transform=transform)
#val_data = datasets.MNIST(root='../input/data',train=False,download=True,transform=transform)
data = CustomImageDataset(img_dir='../lookup_table_head/b/training_data_concatenated/',transform=transform)
train_size = int(len(data)*0.8)
val_size = len(data) - train_size

train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])


train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, prefetch_factor=6, persistent_workers=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=16, prefetch_factor=6, persistent_workers=False)

#model = resnet18(1, 64, activation='leaky_relu').to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
#criterion = nn.BCELoss(reduction='sum')
criterion = nn.MSELoss(reduction='sum')

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    #for i, data in enumerate(dataloader):
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data_cat = data
        data_cat  = data.to(device)
        data_body = data_cat[:,:,:119,:]
        data_tail = data_cat[:,:,119:,:]
        #data = data.view(data.size(0), -1) # This is only applicable for linear model
        optimizer.zero_grad()
        reconstruction_body, mu, logvar = model(data_body)
        data_head = data_body - data_tail
        reconstruction_tail = reconstruction_body - data_head
        bce_loss = criterion(reconstruction_body, data_body) + tail_error_factor*criterion(reconstruction_tail, data_tail)
        loss = final_loss(bce_loss, mu, logvar)
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
            data_cat = data
            data_cat = data.to(device)
            data_body = data_cat[:,:,:119,:]
            data_tail = data_cat[:,:,119:,:]
            #data = data.view(data.size(0), -1) # This is only applicable for linear model
            reconstruction_body, mu, logvar = model(data_body)
            data_head = data_body - data_tail
            reconstruction_tail = reconstruction_body - data_head
            bce_loss = criterion(reconstruction_body, data_body) + tail_error_factor*criterion(reconstruction_tail, data_tail)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data_body.view(batch_size, 1, 119, 119)[:8],
                                  reconstruction_body.view(batch_size, 1, 119, 119)[:8]))
                save_image(both.cpu(), output_dir + "/output_" + str(epoch) + ".png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}",flush=True)
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}",flush=True)
    print(f"Val Loss: {val_epoch_loss:.4f}",flush=True)
plt.plot(train_loss, color='green')
plt.plot(val_loss, color='red')
plt.savefig(output_dir + '/loss.png')
