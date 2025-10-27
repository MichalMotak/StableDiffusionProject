#!/usr/bin/env python
# coding=utf-8

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)
from train_controlnet_my import DatasetRetina, DatasetCADIS  # reuse your dataset definition

from PIL import Image
import torch.nn.functional as F



@torch.no_grad()
def evaluate_manual(
    checkpoint_path,
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    output_dir="results/eval_manual",
    num_samples=4,
    num_inference_steps=30,
    device="cuda",
    image_size=256,
):
    """
    Evaluate ControlNet manually on segmentation-based conditioning (multi-channel).

    Args:
        checkpoint_path (str): Path to the trained ControlNet checkpoint folder (e.g., ".../checkpoint-200000/controlnet").
        pretrained_model_name_or_path (str): Stable Diffusion base model name or directory.
        output_dir (str): Directory where generated images will be saved.
        num_samples (int): Number of random samples from dataset to generate.
        num_inference_steps (int): Diffusion steps for generation.
        device (str): CUDA or CPU device.
        image_size (int): Image resolution (should match training).
    """

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        # print("d")

    else:
        if len(os.listdir(args.output_dir)) != 0:
            print(f'Folder {args.output_dir} is not empty')
            return 0


    # ---------------------------
    # 1️⃣ Load models
    # ---------------------------
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("[INFO] Loading pretrained components...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device, dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device, dtype=torch_dtype)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to(device, dtype=torch_dtype)
    controlnet = ControlNetModel.from_pretrained(checkpoint_path, torch_dtype=torch_dtype).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    vae.eval()
    unet.eval()
    controlnet.eval()
    text_encoder.eval()

    print(vae.config.scaling_factor)


    # ---------------------------
    # 2️⃣ Load dataset
    # ---------------------------
    # dataset = DatasetRetina(image_size=image_size, augmentation_factor=1)
    # dataset = DatasetCADIS(image_size=image_size, augment = False, augmentation_factor=1)

    dataset = DatasetRetina(image_size=256, augment = False, augmentation_factor=1, 
                                from_file=['/home/MichalMo/projects/ControlNet-diffusers/cataract_fold0/cataract_fold0_test.csv'])


    # random.seed(10)
    # indices = random.sample(range(len(dataset)), len(dataset))
    # samples = [dataset[i] for i in indices]

    random.seed(10)
    if num_samples > 0:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = [i for i in range(len(dataset))]

    samples = [dataset[i] for i in indices]


    # num_samples = len(dataset)


    org_image_paths = []

    # ---------------------------
    # 3️⃣ Diffusion sampling loop
    # ---------------------------
    print(f"[INFO] Generating {num_samples} samples...")
    for idx, sample in enumerate(samples):
        prompt = sample["text"]
        cond = sample["mask"].unsqueeze(0).to(device, dtype=torch_dtype)  # [1,14,H,W]

        print(sample["file_name"])
        org_image_paths.append(sample["file_name"])


        xx = sample["file_name"].split("/")
        output_name = f"{xx[-3]}_{xx[-1]}"

        # Tokenize prompt → CLIP hidden states
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = text_encoder(text_inputs.input_ids)[0]  # [1,77,768]

        # Initialize random latent (the "image" in latent space)
        latents = torch.randn(
            (1, unet.in_channels, image_size // 8, image_size // 8),
            device=device,
            dtype=torch_dtype,
        )

        # Prepare scheduler
        noise_scheduler.set_timesteps(num_inference_steps)
        print(f"[INFO] Sampling with {num_inference_steps} steps...")

        for t in tqdm(noise_scheduler.timesteps, desc=f"Sample {idx+1}/{num_samples}"):
            # Predict noise using ControlNet + UNet
            down_block_res_samples, mid_block_res_sample = controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=cond,
                return_dict=False,
            )

            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=torch_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=torch_dtype),
                return_dict=False,
            )[0]

            # Step with scheduler
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # ---------------------------
        # 4️⃣ Decode latents → image
        # ---------------------------
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample  # [1,3,H,W]

        print(image.shape, image.dtype, torch.min(image), torch.max(image))

        # Map from [-1,1] → [0,1]
        # image = (image.clamp(-1, 1) + 1) / 2.0
        image = (image / 2 + 0.5).clamp(0, 1)
        print(image.shape, image.dtype, torch.min(image), torch.max(image))

        image = image.float()

        print(image.shape, image.dtype, torch.min(image), torch.max(image))
        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        print(image.shape, image.dtype, torch.min(image), torch.max(image))

        # Save result
        save_path = os.path.join(output_dir, f"sample_{idx:03d}_{output_name}")
        save_image(image.cpu(), save_path)


        # cond_label_map = np.argmax(cond.cpu().detach().numpy(), axis=1).astype(np.float32)

        # cond_label_map = cond_label_map.squeeze(0)



        # Convert to grayscale (class indices)
        grayscale_mask = np.argmax(cond.cpu().numpy()[0], axis=0).astype(np.uint8)  # shape [256, 256]

        # Optional: add channel dimension
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)  # shape [256, 256, 1]

        print(grayscale_mask.shape)  # (256, 256, 1)
        print(grayscale_mask.min(), grayscale_mask.max())  # 0 13

        img = Image.fromarray(grayscale_mask.astype(np.uint8)[:,:,0], mode = "L")

        save_path_cond = os.path.join(output_dir, f"sample_{idx:03d}_cond_{output_name}")
        # save_image(grayscale_mask, save_path_cond)


        img.save(save_path_cond)



        # save org image

        img_org = (sample["image"].cpu().numpy().transpose(1,2,0)*255.0).astype(np.uint8)
        print(img_org.shape, img_org.dtype, np.min(img_org), np.max(img_org))
        img_org_pil = Image.fromarray(img_org)

        save_path_org_image = os.path.join(output_dir, f"sample_{idx:03d}_org_{output_name}")
        img_org_pil.save(save_path_org_image)



        # Metrics

        # mse_loss = F.mse_loss(torch.transpose(image.squeeze(0), 1,2,0), np.asarray(img_org), reduction="mean")
        image = image.squeeze(0).permute(1, 2, 0).cpu()

        img_org = torch.tensor(img_org).cpu().to(dtype=torch.float32)
        print("METRICS")
        print(image.shape, image.dtype, torch.min(image), torch.max(image))

        print(img_org.shape, img_org.dtype, torch.min(image), torch.max(image))

        mse_loss = F.mse_loss(image, img_org, reduction="mean")

        print("mse_loss ", mse_loss)



        with open(output_dir + '/org_image_paths.txt', 'w') as f:
            for line in org_image_paths:
                f.write(f"{line}\n")




        print(f"[SAVED] {save_path}")





    print(f"\n✅ DONE — {num_samples} images saved in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manual evaluation for multi-channel ControlNet")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to ControlNet checkpoint folder")
    parser.add_argument("--output_dir", type=str, default="results/eval_manual_cataract_mysplit_1_121260_i30")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=256)
    # parser.add_argument("--num_channels", type=int, default=3)

    args = parser.parse_args()



    evaluate_manual(
        checkpoint_path="/home/MichalMo/projects/ControlNet-diffusers/cataract_mysplit_1/checkpoint-121260/controlnet",
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        image_size=args.image_size,
    )
