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

# Define a transformation to apply to the images
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256
    transforms.ToTensor(),          # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Example data (replace with your own data)
data = [
    ("A beautiful sunset over the mountains.", "image1.jpg"),
    ("A cute puppy playing in the grass.", "image2.jpg"),
    # Add more samples as needed
]

# Create an instance of the custom dataset
dataset = TextImageDataset(data, transform=image_transform)

batch_size = 32
shuffle = True  # Shuffle the data during training
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

for texts, images in data_loader:

    print("Textual descriptions:", texts)
    print("Image batch shape:", images.shape)
    break  



