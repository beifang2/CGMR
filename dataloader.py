import torchvision.transforms as tfs
import os
from PIL import Image
from torch.utils import data
import numpy as np
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import cv2

from osgeo import gdal

class Dataset(data.Dataset):
    def __init__(self,path_root="../data/", mode="train", client_name="austin"):
        super(Dataset,self).__init__()
        self.path_root = os.path.join(path_root + mode,client_name)
        self.rs_images_dir = os.listdir(os.path.join(self.path_root, "image"))
        self.rs_images = [os.path.join(self.path_root, "image", img) for img in self.rs_images_dir]
        self.gt_images_dir = os.listdir(os.path.join(self.path_root,"label"))
        self.gt_images = [os.path.join(self.path_root,"label",img) for img in self.rs_images_dir]


    def __getitem__(self, item):
        img = gdal.Open(self.rs_images[item])
        label = gdal.Open(self.gt_images[item])
        img = img.ReadAsArray().transpose(1, 2, 0)
        label = label.ReadAsArray()

        img = img / 255.0
        label = label / 255.0

        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

    def __len__(self):
        return len(self.rs_images)

def load_data_treetype(npz_file, batch_size=64, shuffle=True, num_workers=0):
    # 加载数据
    data = np.load(npz_file)
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']

    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)  # [B, C, H, W]
    y_train = torch.tensor(y_train, dtype=torch.long)


    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader


def normalize_data(x):
    rgb_channels = x[:, :3, :, :]
    rgb_channels = rgb_channels / 255.0

    dsm_channel = x[:, 3, :, :]
    dsm_min, dsm_max = dsm_channel.min(), dsm_channel.max()
    dsm_channel = (dsm_channel - dsm_min) / (dsm_max - dsm_min)

    # 将归一化后的 RGB 和 DSM 重新合并
    x_normalized = torch.cat((rgb_channels, dsm_channel.unsqueeze(1)), dim=1)
    return x_normalized

def load_data_treetype_norm(npz_file, batch_size=64, shuffle=True, num_workers=0):
    data = np.load(npz_file)
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']

    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val = torch.tensor(y_val, dtype=torch.long)

    x_train = normalize_data(x_train)
    x_val = normalize_data(x_val)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    # 封装为 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader


def load_data_treetype_rgb(npz_file, batch_size=64, shuffle=True, num_workers=0):
    # 加载数据
    data = np.load(npz_file)
    x_train, y_train = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']

    x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train, dtype=torch.long)

    x_val = torch.tensor(x_val, dtype=torch.float32).permute(0, 3, 1, 2)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader

class RGBLiDAR_Dataset(Dataset):
    def __init__(self, base_dir, lidar_norm=True):
        self.base_dir = base_dir
        self.lidar_norm = lidar_norm

        self.rgb_dir = os.path.join(base_dir, "images")
        self.lidar_dir = os.path.join(base_dir, "dsm")
        self.lbl_dir = os.path.join(base_dir, "masks")

        if not os.path.exists(self.rgb_dir) or not os.path.exists(self.lidar_dir) or not os.path.exists(self.lbl_dir):
            raise FileNotFoundError(f"One or more directories are missing in the base folder: {base_dir}")

        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.lower().endswith(self.img_extensions)])
        self.lidar_files = sorted([f for f in os.listdir(self.lidar_dir) if f.lower().endswith(self.img_extensions)])
        self.lbl_files = sorted([f for f in os.listdir(self.lbl_dir) if f.lower().endswith(self.img_extensions)])

        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

    def __getitem__(self, idx):
        rgb_image = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[idx]))[:, :, ::-1]  # BGR -> RGB
        lidar_image = cv2.imread(os.path.join(self.lidar_dir, self.lidar_files[idx]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        label_image = cv2.imread(os.path.join(self.lbl_dir, self.lbl_files[idx]), cv2.IMREAD_GRAYSCALE)

        if self.lidar_norm:
            min_v, max_v = lidar_image.min(), lidar_image.max()
            if max_v > min_v:
                lidar_image = (lidar_image - min_v) / (max_v - min_v + 1e-8)
            else:
                lidar_image = np.zeros_like(lidar_image, dtype=np.float32)

        lidar_image = np.expand_dims(lidar_image, axis=-1)

        rgb_image = rgb_image.astype(np.float32) / 255.0

        rgb_image = (rgb_image - self.rgb_mean) / self.rgb_std

        input_image = np.concatenate([rgb_image, lidar_image], axis=-1)  # (H, W, 4)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).float()
        label_image = torch.tensor(label_image, dtype=torch.long)

        return input_image, label_image

    def __len__(self):
        return len(self.rgb_files)

class Dataset_Central(data.Dataset):
    def __init__(self, path_root="../data/", mode="train"):
        super(data.Dataset, self).__init__()
        subfolders = [name for name in os.listdir(path_root + mode)
                      if os.path.isdir(os.path.join(path_root + mode, name))]
        self.rs_images = []
        self.gt_images = []
        for client_name in subfolders:
            self.path_root = os.path.join(path_root + mode, client_name)
            self.rs_images_dir = os.listdir(os.path.join(self.path_root, "image"))
            self.rs_images += [os.path.join(self.path_root, "image", img) for img in self.rs_images_dir]
            self.gt_images_dir = os.listdir(os.path.join(self.path_root, "label"))
            self.gt_images += [os.path.join(self.path_root, "label", img) for img in self.rs_images_dir]

    def __len__(self):
        return len(self.rs_images)

    def __getitem__(self, item):
        img = gdal.Open(self.rs_images[item])
        label = gdal.Open(self.gt_images[item])
        img = img.ReadAsArray().transpose(1, 2, 0)
        label = label.ReadAsArray()

        img = img / 255.0
        label = label / 255.0

        img = tfs.ToTensor()(img)
        label = tfs.ToTensor()(label)

        return img, label

        def __len__(self):
            return len(self.rs_images)
