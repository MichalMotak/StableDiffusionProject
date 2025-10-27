#!/usr/bin/env python
# coding=utf-8

import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from train_controlnet_my import DatasetRetina   # reuse your dataset definition
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ==========================================
# Utility: convert one-hot segmentation mask -> RGB for visualization/ControlNet input
# ==========================================
def onehot_to_rgb(onehot_mask):
    palette = np.array([
        [0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0],
        [255,0,255], [0,255,255], [128,0,0], [0,128,0], [0,0,128],
        [128,128,0], [128,0,128], [0,128,128], [255,128,0]
    ], dtype=np.uint8)

    class_mask = onehot_mask.argmax(0)  # [H, W]
    rgb = palette[class_mask]
    return Image.fromarray(rgb)





@torch.no_grad()
def generate_images_from_batch(
    batch, vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, device, weight_dtype, num_steps=30
):
    """
    Run ControlNet-guided diffusion manually, without the HF pipeline.
    """

    # --- 1. Tokenize text
    text_inputs = tokenizer(
        batch["text"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # print("text_inputs ", text_inputs.size)

    encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]
    print("encoder_hidden_states ", encoder_hidden_states.shape)

    # --- 2. Prepare control condition (segmentation map)
    controlnet_image = batch["mask"].to(device=device, dtype=weight_dtype)
    print("controlnet_image ", controlnet_image.shape)


    # Match latent spatial size (1/8 of image)
    controlnet_image = F.interpolate(
        controlnet_image,
        size=(batch["image"].shape[-2] // 8, batch["image"].shape[-1] // 8),
        mode="nearest"
    )

    # --- 3. Initialize random latent noise
    latents = torch.randn(
        (batch["image"].shape[0], unet.in_channels, batch["image"].shape[-2] // 8, batch["image"].shape[-1] // 8),
        device=device,
        dtype=weight_dtype
    )

    # --- 4. Diffusion denoising loop
    noise_scheduler.set_timesteps(num_steps, device=device)

    for t in tqdm(noise_scheduler.timesteps):
        # predict residual using ControlNet
        down_block_res_samples, mid_block_res_sample = controlnet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False
        )

        noise_pred = unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False
        )[0]

        # Update latents with the scheduler step
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # --- 5. Decode latents back to image space
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2.0  # [0,1] range

    return images







# ==========================================
# Main evaluation function
# ==========================================
def evaluate(
    checkpoint_path,
    pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
    output_dir="results/eval",
    num_samples=4,
    device="cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ------------------------------
    # Load pretrained components
    # ------------------------------
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device, dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device, dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device, dtype=torch_dtype)


    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # ------------------------------
    # Load ControlNet from checkpoint
    # ------------------------------
    print(f"[INFO] Loading ControlNet from checkpoint: {checkpoint_path}")
    controlnet = ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch_dtype).to(device)


    # ------------------------------
    # Prepare dataset
    # ------------------------------
    dataset = DatasetRetina(image_size=256, augmentation_factor=1)

    # Pick random samples
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]




    vae.eval(); text_encoder.eval(); unet.eval(); controlnet.eval()

    batch = next(iter(dataset))  # get some samples
    images = generate_images_from_batch(
        batch=batch,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        device="cuda",
        weight_dtype=torch.float16
    )

    print(images.shape)

    # # Save output
    # save_path = os.path.join(output_dir, f"eval_{i:03d}.png")
    # image.save(save_path)
    # print(f"[SAVED] {save_path}")

    # # Also save conditioning RGB for reference
    # cond_path = os.path.join(output_dir, f"mask_{i:03d}.png")
    # mask.save(cond_path)

    print(f"[DONE] Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ControlNet on segmentation data")
    parser.add_argument("--checkpoint_path", type=str, required=False, default = " ", help="Path to ControlNet checkpoint folder")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="results/", help="Where to save images")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()

    evaluate(
        checkpoint_path="/home/MichalMo/projects/ControlNet-diffusers/controlnet-model/checkpoint-56163/controlnet",
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
