"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import os
from torchvision.utils import make_grid, save_image
from matplotlib.pyplot import errorbar


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        self.root_dir = path_to_folder

        self.image_paths = []
        self.labels = []

        for name in os.listdir(path_to_folder):
            name_folder = os.path.join(path_to_folder, name)
            for image in os.listdir(name_folder):
                image_file = os.path.join(name_folder, image)
                self.image_paths.append(image_file)
                self.labels.append(name_folder)

        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        image_file = self.image_paths[index]
        with Image.open(image_file, 'r') as img:
            return self.transform(img), self.labels[index]

        
if __name__ == '__main__':

    default_path_to_folder = '/mnt/c/Users/Laurine/Documents/DTU Python/Machine Learning Operations/lfw-deepfunneled'
    batch_size = 512
    #default_path_to_folder = '/mnt/c/Users/Laurine/Documents/DTU Python/Machine Learning Operations/lfw-deepfunneled-light'
    #batch_size = 5

    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default=default_path_to_folder, type=str)
    parser.add_argument('-num_workers', default=2, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    # Note we need a high batch size to see an effect of using many
    # number of workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers)
    
    if args.visualize_batch:
        # TODO: visualize a batch of images
        images, labels = next(iter(dataloader))
        grid = make_grid(images)
        save_image(grid, 'batch_visualization.jpg')

        
    if args.get_timing:
        # lets do so repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > 100:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        mean, std = np.mean(res), np.std(res)
        print(f'Timing: {mean}+-{std}')
        
        # save timing result
        with open("timing.csv", "a") as f:
            f.write(f"{args.num_workers};{mean};{std}\n")

