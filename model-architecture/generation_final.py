import torch
from diffusion import Diffusion
import model_loader
from ddpm import DDPMSampler
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
from tqdm import tqdm

WIDTH = 128
HEIGHT = 128
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

strength = 0.9
sampler = "ddpm"
num_inference_steps = 25
seed = 42

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "./data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

#diffusion = models['diffusion']
diffusion = Diffusion()
diffusion.to(DEVICE)

# Load the weights
checkpoint = torch.load('./checkpoint_epoch_2_0.pt', map_location=DEVICE)

#diffusion_state_dict = diffusion.load_state_dict(checkpoint)

print(type(diffusion))
# Print model's state_dict
#print("Model's state_dict:")
#for param_tensor in diffusion_state_dict:
    #  print(param_tensor, "\t", diffusion_state_dict[param_tensor].size())
#    print(param_tensor)
#    break
# models['diffusion'] = diffusion_state_dict

prompt = "Give me a blue circular logo for a healthcare brand conveying saftey and service."

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
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def generate(
    prompt,
    strength=0.9,
    sampler_name="ddpm",
    n_inference_steps=25,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        # Convert into a list of length Seq_Len=77
        tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=77
        ).input_ids
        # (Batch_Size, Seq_Len)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        context = clip(tokens)
        to_idle(clip)

        # if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_timesteps(n_inference_steps)
        # else:
        #   raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # (Batch_Size, 4, Latents_Height, Latents_Width)
        latents = torch.randn(latents_shape, generator=generator, device=device)

        # diffusion = models["diffusion"]
        # diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            # if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                # model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            # if do_cfg:
            #     output_cond, output_uncond = model_output.chunk(2)
            #     model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def main():
    output_image = generate(
        prompt=prompt,
        strength=strength,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device=None,
        tokenizer=tokenizer,
    )

    final_image = Image.fromarray(output_image)

    final_image.save("output_image.jpg")
if __name__=="__main__":
    main()
