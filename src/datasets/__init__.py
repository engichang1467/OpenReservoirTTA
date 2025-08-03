from .cifar100c_loader import CIFAR100CDataset, download_cifar100c, get_all_cifar100c_corruptions

def get_cifar100c_dataset(data_dir, corruption, severity, transform=None):
    download_cifar100c(data_dir)
    return CIFAR100CDataset(data_dir, corruption, severity, transform)
