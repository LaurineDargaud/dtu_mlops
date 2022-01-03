import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def mnist():
    # exchange with the corrupted mnist dataset
    
    path = 'C:/Users/Laurine/Documents/DTU Python/Machine Learning Operations/dtu_mlops/data/corruptmnist/'
    
    ### get TRAIN dataloader
    all_torch_images, all_torch_labels = [], []
    for i in range (5):
        file_path = path+'train_{}.npz'.format(i)
        np_array = np.load(file_path)
        all_torch_images.append(torch.from_numpy(np_array['images']))
        all_torch_labels.append(torch.from_numpy(np_array['labels']))
    torch_images= torch.cat(all_torch_images, 0)
    torch_labels = torch.cat(all_torch_labels, 0)
    train_dataset = TensorDataset(torch_images, torch_labels)
    train_loader = DataLoader(train_dataset)
    
    ### get TEST dataloader
    file_path = path+'test.npz'
    np_array = np.load(file_path)
    torch_test_images = torch.from_numpy(np_array['images'])
    torch_test_labels = torch.from_numpy(np_array['labels'])
    test_dataset= TensorDataset(torch_test_images, torch_test_labels)
    test_loader = DataLoader(test_dataset)
    
    return train_loader, test_loader
