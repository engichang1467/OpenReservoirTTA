import os
import urllib.request
import tarfile
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
CIFAR100C_FILENAME = "CIFAR-100-C.tar"
CIFAR100C_DIRNAME = "CIFAR-100-C"

# def get_cifar100c_dataset(data_dir, corruption, severity, transform=None):
#     download_cifar100c(data_dir)
#     return CIFAR100CDataset(data_dir, corruption, severity, transform)


def get_all_cifar100c_corruptions():
    return [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
        'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise',
        'gaussian_blur', 'spatter', 'saturate'
    ]

def download_cifar100c(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, CIFAR100C_FILENAME)
    if not os.path.exists(os.path.join(data_dir, CIFAR100C_DIRNAME)):
        if not os.path.exists(filepath):
            print("Downloading CIFAR-100-C...")
            urllib.request.urlretrieve(CIFAR100C_URL, filepath)
        print("Extracting CIFAR-100-C...")
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(path=data_dir)
    else:
        print("CIFAR-100-C already exists.")



class CIFAR100CDataset(Dataset):
    def __init__(self, data_dir, corruption, severity=1, transform=None):
        """
        corruption: e.g. 'gaussian_noise', 'motion_blur'
        severity: 1–5
        """
        self.corruption = corruption
        self.severity = severity
        self.transform = transform or transforms.ToTensor()
        self.data_path = os.path.join(data_dir, CIFAR100C_DIRNAME)
        self._load_data()

    def _load_data(self):
        images_path = os.path.join(self.data_path, f'{self.corruption}.npy')
        labels_path = os.path.join(self.data_path, 'labels.npy')
        all_images = np.load(images_path)
        labels = np.load(labels_path)
        
        # Select the right severity slice
        start = (self.severity - 1) * 10000
        end = self.severity * 10000
        self.images = all_images[start:end]
        self.labels = labels[start:end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])
