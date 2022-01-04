import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def mnist():
    # exchange with the corrupted mnist dataset
    
    path = '/mnt/c/Users/Laurine/Documents/DTU Python/Machine Learning Operations/dtu_mlops/data/corruptmnist/'
    
    ### get TRAIN dataloader
    all_torch_images, all_torch_labels = [], []
    for i in range (5):
        file_path = path+'train_{}.npz'.format(i)
        np_array = np.load(file_path)
        all_torch_images.append(torch.from_numpy(np_array['images']))
        all_torch_labels.append(torch.from_numpy(np_array['labels']))
    torch_images= torch.cat(all_torch_images, 0)
    torch_labels = torch.cat(all_torch_labels, 0)
    torch_images, torch_labels = torch_images.type(torch.FloatTensor), torch_labels.type(torch.LongTensor)
    train_dataset = TensorDataset(torch_images, torch_labels)
    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    
    ### get TEST dataloader
    file_path = path+'test.npz'
    np_array = np.load(file_path)
    torch_test_images = torch.from_numpy(np_array['images'])
    torch_test_labels = torch.from_numpy(np_array['labels'])
    torch_test_images, torch_test_labels = torch_test_images.type(torch.FloatTensor), torch_test_labels.type(torch.LongTensor)
    test_dataset= TensorDataset(torch_test_images, torch_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    return train_loader, test_loader
