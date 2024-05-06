import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


### Defining Data Loader For Training 
class TextImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the textual description and image path from the data
        text, image_path = self.data[idx]
        
        # Open the image using PIL
        image = Image.open(image_path).convert("RGB")
        
        # Apply any transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Return the textual description and the image
        return text, image





