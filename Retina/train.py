# train_controlnet_depth_only.py
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from datasets import load_dataset
from tqdm import tqdm

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    DDPMScheduler,
    AutoencoderKL
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator





from test_gpu import setup_cuda


import pandas as pd


# -------------------------
# Config (edit these)
# -------------------------
dataset_csv = "/home/MichalMo/projects/StableDiffusion/Retina/dataset/cadis/pairs.csv"   # produced earlier (columns: image_path, depth_path, mask_path, text)
pretrained_sd = "runwayml/stable-diffusion-v1-5"   # base SD checkpoint
controlnet_pretrained = "lllyasviel/sd-controlnet-depth"  # controlnet depth initial weights
output_dir = "./test2"
learning_rate = 1e-5
train_batch_size = 32
num_train_epochs = 500
mixed_precision = "fp16"
gradient_accumulation_steps = 1
save_every_steps = 1000
seed = 42
image_size = 128
# device = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------

torch.manual_seed(seed)

# ---------- Dataset ----------
class PairedImageDataset(Dataset):
    def __init__(self, csv_file, image_size=512):
        self.df = pd.read_csv(csv_file)
        self.size = image_size
        self.image_trans = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
        # normalize to [-1,1] for VAE images
        self.normalize = T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        depth = Image.open(row["depth_path"]).convert("RGB")   # convert depth to 3-channel
        #mask = Image.open(row["mask_path"]).convert("RGB")     # convert mask to 3-channel

        image = self.normalize(self.image_trans(image))
        depth = self.normalize(self.image_trans(depth))
        #mask = self.normalize(self.image_trans(mask))

        # return dict of tensors
        return {
            "image": image,        # target image (to denoise toward)
            "depth": depth,        # control image #1
            #"mask": mask,          # control image #2 (optional)
            "text": ""             # empty prompt
        }

# ---------- Load models ----------


setup_cuda(use_memory_fraction=0.8, num_threads=16, visible_devices="0,1", multiGPU=True)

# setup_cuda(use_memory_fraction=0.6, num_threads=16, visible_devices="0,1", use_cuda_with_id = 0)




# NOTE: We will freeze the SD components and train ControlNet weights only.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

vae = AutoencoderKL.from_pretrained(pretrained_sd, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(pretrained_sd, subfolder="unet")

# load ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_pretrained)

# Freeze VAE, UNet, text_encoder (we will only train controlnet)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
for param in vae.parameters(): param.requires_grad = False
for param in unet.parameters(): param.requires_grad = False
for param in text_encoder.parameters(): param.requires_grad = False

# confirm trainable params are in controlnet only
for name, p in controlnet.named_parameters():
    # leave them trainable
    pass

# Build a pipeline (we won't use the pipeline for training step-by-step,
# but it is useful for later inference)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    pretrained_sd,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16 if mixed_precision=="fp16" else torch.float32
)

# ---------- Dataloader ----------
ds = PairedImageDataset(dataset_csv, image_size=image_size)
loader = DataLoader(ds, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Optimizer ----------
optimizer = torch.optim.AdamW(controlnet.parameters(), lr=learning_rate)

# ---------- Scheduler (optional) ----------
from transformers import get_scheduler
num_update_steps_per_epoch = len(loader) // gradient_accumulation_steps
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# ---------- Accelerator ----------
accelerator = Accelerator(mixed_precision=mixed_precision)
print(accelerator.device)
controlnet, optimizer, loader, lr_scheduler = accelerator.prepare(controlnet, optimizer, loader, lr_scheduler)

device = accelerator.device

print(device)

vae= vae.to(device)
text_encoder = text_encoder.to(device)
unet = unet.to(device)

# ---------- Noise scheduler ----------
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_sd, subfolder="scheduler")

# ---------- Training loop ----------
from torch.nn import functional as F

global_step = 0
for epoch in range(num_train_epochs):
    controlnet.train()
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        # move to device
        images = batch["image"].to(accelerator.device, dtype=torch.float32)
        depth_images = batch["depth"].to(accelerator.device, dtype=torch.float32).cuda()
        #mask_images = batch["mask"].to(accelerator.device, dtype=torch.float32).cuda()
        texts = batch["text"]  # all ""

        # print(images.device, depth_images.device)
        # print(images.shape, depth_images.shape)

        # 1) encode images to latents (VAE)
        with torch.no_grad():
            latents = vae.encode((images + 1) / 2.0).latent_dist.sample() * vae.config.scaling_factor

        # print(latents.shape)

        # 2) Sample noise and timesteps
        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()

        # add noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 3) get text embeddings (empty prompt -> identical embedding for all)
        # We simply tokenize empty strings (cached if you want speed)
        text_inputs = tokenizer(list(texts), padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = text_inputs["input_ids"].to(accelerator.device)
        # get text embeddings (we won't train text encoder)
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        # 4) ControlNet forward: prepare control images as required
        # controlnet in diffusers expects 3-channel images, float in [-1,1] scaled
        # We pass depth_images as the conditioning_image, and optionally mask_images as a second control
        # If you want only depth, comment out mask usage.
        controlnet_cond = depth_images  # shape [B,3,H,W]
        # If training multi-controlnet: you'd have two controlnet models and combine their outputs; more code needed.

        # 5) predict noise residual: pass through UNet together with controlnet output
        # In the ControlNet training flow, we compute "controlnet_down_sample" outputs and pass them to UNet.
        # Simplify by calling controlnet to get the added residuals via "controlnet" forward.
        # NOTE: the exact internals in diffusers examples call controlnet(..., return_dict=True) -> controlnet_output
        down_block_res_samples, mid_block_res_sample = controlnet(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond, return_dict=False)



        # Now call UNet with additional residuals from controlnet:
        # The diffusers architecture expects the controlnet added conditioning to be merged into unet's forward.
        unet_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[
                            sample.to(dtype=torch.float32) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float32),
                        return_dict=False)[0]
        
        # The above line is conceptual — if the exact unet signature in your diffusers version differs,
        # check the diffusers training/controlnet example. The typical pattern is:
        # controlnet_out = controlnet(...); unet_out = unet(..., controlnet_cond=controlnet_out)
        # If your diffusers version does not accept controlnet_cond parameter, follow the official example code.

        # 6) compute loss between predicted noise and true noise
        loss = F.mse_loss(unet_pred, noise)

        # backprop
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % 100 == 0:
            accelerator.print(f"Step {global_step} loss {loss.item():.4f}")

        # checkpoint
        if global_step % save_every_steps == 0:
            if accelerator.is_main_process:
                controlnet_to_save = accelerator.unwrap_model(controlnet)
                controlnet_to_save.save_pretrained(os.path.join(output_dir, f"controlnet_step_{global_step}"))

    # # epoch end save
    # if accelerator.is_main_process:
    #     controlnet_to_save = accelerator.unwrap_model(controlnet)
    #     controlnet_to_save.save_pretrained(os.path.join(output_dir, f"controlnet_epoch_{epoch}"))

print("Training finished. Final weights saved to", output_dir)
