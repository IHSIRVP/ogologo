import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from clip import CLIP
import model_converter
from transformers import CLIPTokenizer
# import model_loader
# from ddpm import DDPMSampler
from ddpm_cos import DDPMSampler
import csv
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
# from transformers import CLIPTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.autograd.set_detect_anomaly(True)

# Set hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 1
LEARNING_RATE = 5e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)

WIDTH = 128
HEIGHT = 128
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

class TextImageDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_seq_length=77, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with text descriptions and image file names.
            img_dir (str): Directory where the images are stored.
            tokenizer (function): Tokenizer function to encode text descriptions.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file,  sep='^([^,]+),', engine='python', usecols=['Image Path', 'Description'], quoting=csv.QUOTE_MINIMAL, nrows=500)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the image
        img_name = self.data.iloc[idx, 0]
        img_path = f"{img_name}"
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Tokenize the text description
        description = self.data.iloc[idx, 1]
        text_tokens = self.tokenizer(description, padding='max_length', max_length=self.max_seq_length, truncation=True, return_tensors="pt")

        # Remove extra batch dimension added by return_tensors
        text_tokens = text_tokens.squeeze(0)

        return image, text_tokens

# tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")

# Define a transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),  
    transforms.ToTensor(),          
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize 
])

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]# temporal scaling
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def main():
    ckpt_path = "./data/v1-5-pruned-emaonly.ckpt"
    state_dict = model_converter.load_from_standard_weights(ckpt_path, DEVICE)
    print("Loading data")
    # Load the data from a CSV file into a DataLoader
    train_data = TextImageDataset(csv_file="../image_descriptions.csv", tokenizer=tokenizer.encode,  max_seq_length=77, transform=transform)  # Assumes dataset is defined
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize models
    encoder = VAE_Encoder().to(DEVICE)
    encoder.load_state_dict(state_dict['encoder'], strict=True)
    decoder = VAE_Decoder().to(DEVICE)
    decoder.load_state_dict(state_dict['decoder'], strict=True)
    diffusion = Diffusion().to(DEVICE)
    clip = CLIP().to(DEVICE)
    clip.load_state_dict(state_dict['clip'], strict=True)

    # Define optimizers
    optimizers = {
        # 'encoder': AdamW(encoder.parameters(), lr=LEARNING_RATE),
        # 'decoder': AdamW(decoder.parameters(), lr=LEARNING_RATE),
        'diffusion': AdamW(diffusion.parameters(), lr=LEARNING_RATE)
    }

    # Define LR schedulers
    schedulers = {
        # 'encoder': CosineAnnealingLR(optimizers['encoder'], T_max=NUM_EPOCHS),
        # 'decoder': CosineAnnealingLR(optimizers['decoder'], T_max=NUM_EPOCHS),
        'diffusion': CosineAnnealingLR(optimizers['diffusion'], T_max=NUM_EPOCHS)
    }

    scaler = GradScaler()
    epoch_losses = []

    # Training loop
    for epoch in range(NUM_EPOCHS):

        # encoder.train()
        # decoder.train()
        diffusion.train()
        epoch_loss = 0
        batch_losses = []

        # for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):

            images, texts = batch
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            with autocast():
                loss = 0
                # Reset gradients for all optimizers at the start of the batch
                for optimizer in optimizers.values():
                    optimizer.zero_grad()

                # print(f"Images shape: {images.shape}")

                # Encode texts to latents
                # text_tokens = tokenizer.batch_encode_plus(
                #     texts, padding="max_length", max_length=77
                # ).input_ids
                # text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)
                text_tokens = torch.tensor(texts, dtype=torch.long, device=DEVICE)
                context = clip(text_tokens)
                context = context.to(DEVICE)

                generator = torch.Generator(device=DEVICE)
                generator.manual_seed(42)

                sampler = DDPMSampler(generator)
                sampler.set_inference_timesteps(25)

                latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

                input_image_tensor = rescale(images, (0, 255), (-1, 1))
                # print(f"Input image tensor after rescale: {input_image_tensor.shape}")

                encoder_noise = torch.randn(latents_shape, generator=generator, device=DEVICE)
                # print(f"Encoder noise: {encoder_noise.shape}")

                input_image_tensor = input_image_tensor.to(DEVICE)

                latents = encoder(input_image_tensor, encoder_noise)
                # print(f"Latent shape: {latents.shape}")

                latents = latents.to(DEVICE)
                # Add noise to the latents (the encoded input image)
                sampler.set_strength(strength=0.9)
                latents = sampler.add_noise(latents, sampler.timesteps[0])
                latents = latents.to(DEVICE)
                # # Encode images to latents
                # images = images.to(DEVICE)
                # latents = encoder(images)

                # Diffusion step

                timesteps = tqdm(sampler.timesteps)
                for i, timestep in enumerate(timesteps):

                    time_embedding = get_time_embedding(timestep).to(DEVICE)

                    model_input = latents
                    
                    # model_input = model_input.repeat(2,1,1,1)
                    # print("Model input and context shape")
                    # print(model_input.shape, context.shape)

                    # model_output is the predicted noise
                    model_output = diffusion(model_input, context, time_embedding)
                    model_output.to(DEVICE)
                    if True:
                        output_cond, output_uncond = model_output.chunk(2)
                        # model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
                        model_outputs = 8 * (output_cond - output_uncond) + output_uncond
                    # print(model_outputs.shape, 'cfg')
                    # print(latents.shape,'latents')
                    # print(model_output.shape, 'actual omput to sampler') 
                    latents = sampler.step(timestep, latents, model_output)

                    # Loss and optimization
                    diffusion_loss = torch.nn.functional.mse_loss(model_output, encoder_noise)
                    loss += diffusion_loss

                scaler.scale(loss).backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                batch_losses.append(loss.item())
                print(f"Epoch: {epoch}, Loss: {loss}")

        epoch_losses.append(epoch_loss / len(train_loader))
        plt.figure(figsize=(10, 4))
        plt.plot(batch_losses, label=f'Batch Loss Epoch {epoch+1}')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title(f'Loss per Batch in Epoch {epoch+1}')
        plt.legend()
        plt.savefig(f'batch_loss_epoch_cosine_{epoch+1}.png')
        plt.close()
        for scheduler in schedulers.values():
            scheduler.step()

        # Save checkpoints
        # torch.save({...}, f'checkpoint_epoch_{epoch}.pt')
        torch.save(diffusion.state_dict(), f'checkpoint_epoch_2_cosine_{epoch}.pt')
    plt.figure(figsize=(10, 4))
    plt.plot(range(NUM_EPOCHS), epoch_losses, label='Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.savefig('epoch_loss_cosine.png')
    plt.close()
    # torch.save({...}, 'final_model_checkpoint.pt')
    # torch.save({
    #    'model_state_dict': diffusion.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'epoch': epoch,
    #    'loss': loss
    # }, f'final_checkpoint.pt')
    print("Training completed.")

if __name__=="__main__":
    main()
