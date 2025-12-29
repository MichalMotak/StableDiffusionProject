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


from diffusers import StableDiffusionControlNetPipeline

class StableDiffusionControlNetPipelineMultiCond(StableDiffusionControlNetPipeline):
    def check_image(self, image, prompt, prompt_embeds):
        # Skip the 3-channel / batch-size checks
        return image


    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        # Skip all conversions — assume image is already torch tensor [B,14,H,W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image)).permute(2,0,1).unsqueeze(0)
        image = image.to(device=device, dtype=dtype)
        return image





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


    pipe = StableDiffusionControlNetPipelineMultiCond.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    # ------------------------------
    # Prepare dataset
    # ------------------------------
    dataset = DatasetRetina(image_size=256, augmentation_factor=1)


    # Pick random samples
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]

    print(f"[INFO] Generating {num_samples} samples...")
    for i, sample in enumerate(tqdm(samples)):
        prompt = sample["text"]
        mask = sample["mask"].numpy()  # [14,H,W]

        # Convert one-hot mask to RGB

        print(mask.shape)
        mask = np.expand_dims(mask, 0)
        print(mask.shape) # [1,14,256,256]

        # mask = np.ones((1,14,256,256), dtype = np.float16)
        # print(mask.shape)

        # Generate image
        with torch.autocast("cuda"):
            image = pipe(
                prompt=prompt,
                image=mask,
                num_inference_steps=30,
                guidance_scale=9.0,
            ).images[0]

        # Save output
        save_path = os.path.join(output_dir, f"eval_{i:03d}.png")
        image.save(save_path)
        print(f"[SAVED] {save_path}")

        # Also save conditioning RGB for reference
        cond_path = os.path.join(output_dir, f"mask_{i:03d}.png")
        mask.save(cond_path)

    print(f"[DONE] Results saved to {output_dir}")
    print(f"[DONE] Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ControlNet on segmentation data")
    parser.add_argument("--checkpoint_path", type=str, required=False, default = " ", help="Path to ControlNet checkpoint folder")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="results/eval", help="Where to save images")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    args = parser.parse_args()

    evaluate(
        checkpoint_path="/home/MichalMo/projects/ControlNet-diffusers/t4/checkpoint-45120/controlnet",
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )
