import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch


def create_dataframe(directory_path, image_specifics): 
    images = []
    labels = []
    for label in os.listdir(directory_path):
        file_path = os.path.join(directory_path, label)
        if os.path.isdir(file_path):
            file_path =  os.path.join(file_path, image_specifics)
            for imagename in os.listdir(file_path):
                image_path = os.path.join(file_path, imagename)
                if image_path.endswith('.bmp'):
                    image = read_image(image_path)
                    images.append(image)
                    labels.append(label)
    df = pd.DataFrame({'images': images, 'labels': labels})
    return df


def read_image(filepath, target_size=(40,40)):
    # Read the image from the filepath
    img = cv2.imread(filepath)
    # Resize the image to the target size
    img = cv2.resize(img, target_size, interpolation = cv2.INTER_LINEAR)
    # Convert image to RGB (OpenCV reads images in BGR format by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the image to a NumPy array and normalize the pixel values
    img_np = np.array(img, dtype = np.float32) / 255
    # Rearrange the dimensions to (channels, height, width)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

#create multiple dataframes and then merge them. 


class Data(Dataset):
    def __init__(self, X, y, transform=None):
        X = np.array([arr.astype(float) for arr in X.values], dtype=float)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.len = self.X.shape[0]
        self.transform = transform
       
    def __getitem__(self, index):
        image = self.X[index].numpy()#.transpose(1, 2, 0)  # Convert to HWC format
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]
   
    def __len__(self):
        return self.len


class DataModule: 
    def __init__(self, X, y, transform=None):
        self.dataset = Data(X, y, transform=transform)

    def get_dataloader(self, batch_size, num_workers=0):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)

    def val_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)
