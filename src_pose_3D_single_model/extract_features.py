from typing import Dict, Iterable, Callable
import torch
from ResNet_Blocks_3D import resnet18
import torchvision.transforms as transforms
import os
from CustomDataset_images_only import CustomImageDataset
from torch.utils.data import DataLoader
import numpy as np
from torch import nn, Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(3, 12, activation='leaky_relu').to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('resnet_pose_220626.pt'))

date = '220629'
batch_size = 200
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

real_folder_add = '../validation_data_3D_pose_shifted/images_real/'
gray_folder_add = '../validation_data_3D_pose_shifted/images_gray/'

im_files = sorted(os.listdir(real_folder_add))

data = CustomImageDataset(im_files,  real_folder_add, gray_folder_add, transform=transform)

loader = DataLoader(data, batch_size=batch_size,shuffle=False,num_workers=2*12,prefetch_factor=2,persistent_workers=True)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

resnet_features = FeatureExtractor(model, layers=["encoder"])

for idx, data in enumerate(loader):
    im_three_channels_real, im_three_channels_gray = data
    features = resnet_features()

print({name: output.shape for name, output in features.items()})
