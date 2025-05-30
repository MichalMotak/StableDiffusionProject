########################3

'''

Script for training DDPM for 3D Ultrasound dataset

'''

#################################3


import dataclasses
from dataclasses import dataclass, field
from typing import List

import time
from datasets import load_dataset
import matplotlib.pyplot as plt
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
from diffusers import DDPMPipeline
import math
import csv
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from diffusers import UNet2DModel
from torchvision import transforms
import random 
import numpy as np
from diffusers.utils import BaseOutput
import json
from PIL import Image
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as T
from diffusers import UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import LambdaLR


import sys
sys.path.append(os.path.abspath("Fast_DDPM"))
print(sys.path)
print("Current working directory:", os.getcwd())
print("fast_ddpm path in sys.path:", os.path.abspath("Fast_DDPM"))
print("Directory listing:", os.listdir("Fast_DDPM"))
from runners.diffusion import Diffusion

#######################

from test_gpu import setup_cuda





@dataclass
class TrainingConfig:

    image_size: int = 128  # the generated image resolution
    image_format:str = "gray"

    train_batch_size: int  = 32
    eval_batch_size: int  = 8  # how many images to sample during evaluation

    num_epochs: int = 100
    save_image_epochs: int  = 5
    save_model_epochs: int  = 5

    
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 50


    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    results_dir: str = "fast_ddpm_tests" 
    dataset_dir: str = "dataset_Vol3_4"


    cond_dim: int = 6
    cond_embedding_dim: int = 128
    cond_parameters: List = field(default_factory=lambda: ['pos_x', 'pos_y', 'pos_z', "rot_x", "rot_y", "rot_z"])



def get_custom_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.1,  # Final LR will be min_lr_ratio * base LR
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        scaled_decay = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        return max(scaled_decay, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)



class CustomConditionedUNet(nn.Module):
    def __init__(self, base_unet: UNet2DConditionModel, cond_dim, embedding_dim):
        super().__init__()
        self.unet = base_unet

        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.config = self.unet.config
        self.cond_dim = cond_dim
        self.embedding_dim = embedding_dim

    def forward(self, sample, timestep, cond_vector=None):
        cond_emb = self.cond_embed(cond_vector).unsqueeze(1)
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=cond_emb
        )



    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.unet.save_pretrained(os.path.join(path, "unet"))
        torch.save(self.cond_embed.state_dict(), os.path.join(path, "cond_embed.pth"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({
                "cond_dim": self.cond_dim,
                "embedding_dim": self.embedding_dim
            }, f)






    @classmethod
    def from_pretrained(cls, path):
        base_unet = UNet2DConditionModel.from_pretrained(os.path.join(path, "unet"))
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(base_unet, **config)
        model.cond_embed.load_state_dict(torch.load(os.path.join(path, "cond_embed.pth")))
        return model



# class CustomConditionedUNet(nn.Module):
#     def __init__(self, base_unet: UNet2DConditionModel, cond_dim=6, embedding_dim=128):
#         super().__init__()
#         self.unet = base_unet
#         self.cond_embed = nn.Sequential(
#             nn.Linear(cond_dim, embedding_dim),
#             nn.SiLU(),
#             nn.Linear(embedding_dim, embedding_dim)
#         )

#         self.config = self.unet.config



#     def forward(self, sample, timestep, cond_vector=None):
#         # [B, 128] -> [B, 1, 128] (cross-attention format)
#         cond_emb = self.cond_embed(cond_vector).unsqueeze(1)

#         return self.unet(
#             sample=sample,
#             timestep=timestep,
#             encoder_hidden_states=cond_emb
#         )





# class ConditionedDDPMPipeline(DDPMPipeline):
#     def __init__(self, unet, scheduler, image_size):
#         super().__init__(unet=unet, scheduler=scheduler)
#         self.image_size = image_size

#     @torch.no_grad()
#     def __call__(self, cond_vector, batch_size=1, generator=None, num_inference_steps=1000):

#         sample_shape = (batch_size, self.unet.config.in_channels, self.image_size, self.image_size)  # adapt to your model
#         print("sample_shape ", sample_shape)
#         image = torch.randn(sample_shape, generator=generator).to("cuda")

#         # print(cond_vector.size())

#         self.scheduler.set_timesteps(num_inference_steps)
#         for t in self.scheduler.timesteps:
#             model_output = self.unet(
#                 sample=image,
#                 timestep=t,
#                 cond_vector=cond_vector.to("cuda")  # pass conditioning vector
#             )
#             image = self.scheduler.step(model_output.sample, t, image).prev_sample

#         return image

class ConditionedDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, image_size):
        super().__init__(unet=unet, scheduler=scheduler)
        self.image_size = image_size

    @torch.no_grad()
    def __call__(self, cond_vector, batch_size=1, generator=None, num_inference_steps=1000):
        device = next(self.unet.parameters()).device
        cond_vector = cond_vector.to(device)

        sample_shape = (batch_size, self.unet.config.in_channels, self.image_size, self.image_size)  # adapt to your model
        # print("sample_shape ", sample_shape)
        image = torch.randn(sample_shape, generator=generator).to(device)

        # print(cond_vector.size())

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            model_output = self.unet(
                sample=image,
                timestep=t,
                cond_vector=cond_vector  # pass conditioning vector
            )

            image = self.scheduler.step(model_output.sample, t, image).prev_sample

        # return BaseOutput(images=image)
        return image

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        self.unet.save_pretrained(os.path.join(path, "model"))
        self.scheduler.save_pretrained(os.path.join(path, "scheduler"))
        with open(os.path.join(path, "pipeline_config.json"), "w") as f:
            json.dump({"image_size": self.image_size}, f)

    @classmethod
    def from_pretrained(cls, path):
        from_custom_unet = CustomConditionedUNet.from_pretrained(os.path.join(path, "model"))
        scheduler = DDPMScheduler.from_pretrained(os.path.join(path, "scheduler"))
        with open(os.path.join(path, "pipeline_config.json")) as f:
            pipe_cfg = json.load(f)
        return cls(unet=from_custom_unet, scheduler=scheduler, **pipe_cfg)





 

class ImageVectorDataset3DUSG(Dataset):
    def __init__(self, image_dir, label_csv, image_size=64):
        self.df = pd.read_csv(label_csv)
        self.image_dir = image_dir

        # Filter rows based on step from image filenames like image_0.png to image_359.png
        # self.df["index"] = self.df["filename"].str.extract(r"_(\d+)\.")[0].astype(int)
        # self.df = self.df[self.df["index"] % step == 0].reset_index(drop=True)

        # print(self.df)

        self.image_names = os.listdir(self.image_dir)[:1888]#[:18624]
        print("image_names ", len(self.image_names))

        self.df = self.df[self.df['filename'].isin(self.image_names)].reset_index()
        print(self.df.shape)


        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])



        print(self.df.shape)

        for col_name  in ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]:
            # print(col_name)
            d = self.df[col_name].values
            d_u = np.unique(d)
            print(col_name, len(d_u), d_u)
            # diffs = np.diff(x)

            # diffs = np.diff(d_u)
            # if np.all(diffs ==diffs[0]) == True:
            #     # print(diffs)
            #     pass



    def preprocess(self):

        # pos_y_get = np.unique(self.df["pos_y"].values)

        # print(pos_y_get)

        rot_x_get = np.unique(self.df["rot_x"].values)
        rot_y_get = np.unique(self.df["rot_y"].values)
        rot_z_get = np.unique(self.df["rot_z"].values)

        print(len(rot_x_get), rot_x_get)
        print(len(rot_y_get), rot_y_get)
        print(len(rot_z_get), rot_z_get)


        # for get in [rot_x_get, rot_y_get, rot_z_get]
        self.df = self.df[(self.df["rot_x"].isin(rot_x_get)) & (self.df["rot_y"].isin(rot_y_get)) & (self.df["rot_z"].isin(rot_z_get))]
        self.df.reset_index()
        print("After preprocess ", self.df.shape)



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        vector = torch.tensor([row['pos_x'], row['pos_y'], row['pos_z'], row['rot_x'], row['rot_y'], row['rot_z']], dtype=torch.float32)
        return image, vector



    def generate_images_from_df(self, df_sample):
            
        filename_names = df_sample['filename']
        # print(filename_names)
        # print("df_sample ", df_sample)

        images_out = []
        for index, row in df_sample.iterrows():
            # print(index, row)

            image, _ = self.__getitem__(index)
            # print(image.shape)
            images_out.append(image)

        images_out_tensor = torch.stack(images_out, dim=1)
        # print(images_out_tensor.shape)
        return images_out_tensor




def loss_fastddpm(model,
                          x_img: torch.Tensor,
                          x_gt: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, 
                          cond_vector: torch.Tensor,
                          keepdim=False):
    # a: a_T in DDIM
    # 1-a: 1-a_T in DDIM 
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # X_T
    x = x_gt * a.sqrt() + e * (1.0 - a).sqrt()
    # print(type(x_img), type(x))
    # print(x_img.device, x.device, t.device, cond_vector.device)
    # print(torch.cat([x_img, x], dim=1).size())
    output = model(torch.cat([x_img, x], dim=1), t.float(), cond_vector = cond_vector)

    # print(type(output), output.size())
    output = output.sample
    # print(type(output), output.size())

    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



def my_make_grid(images, rows, cols):

    grid = make_grid(images, nrow=3, padding=2, normalize=True)

    grid_pil = to_pil_image(grid)
    return grid_pil

def mse(img1, img2):

    # print(img1.size(), img2.size())

    mse_list = []
    for i in range(img1.size()[0]):

        v = F.mse_loss(img1[i], img2[i]).item()
        # print(v)
        mse_list.append(v)

    return torch.Tensor(mse_list)


def evaluate(config, epoch, pipeline, dfs, gt_images):

    # dfs = df.sample(config.eval_batch_size)
    # c = torch.Tensor(dfs[config.cond_parameters].values)
    # print(c, c.size())
    cond_vector = torch.Tensor(dfs[config.cond_parameters].values)

    # filename_names = dfs['filename']
    # print(filename_names)


    images = pipeline(
                    cond_vector=torch.tensor(cond_vector, dtype = torch.half), # .unsqueeze(2)
                    batch_size=config.eval_batch_size)
    


    # print(images.shape, gt_images.shape)
    # print(images.device, gt_images[0].device)
    error = mse(images[:,0,:,:], gt_images[0].to(images.device))
    # print(error.size())
    
    # print(error.mean())


    image_grid = my_make_grid(images,rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.results_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

    return error




def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): Number of epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f"Initial validation loss: {val_loss:.4f}")

        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epoch(s)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")




def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(x) + 1)
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule =='alpha_cosine':
        s = 0.008
        timesteps = np.arange(0, num_diffusion_timesteps+1, dtype=np.float64)/num_diffusion_timesteps
        alphas_cumprod = np.cos((timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == 'alpha_sigmoid':
        x = np.linspace(-6, 6, 1001)
        alphas_cumprod = sigmoid(x)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)
    elif beta_schedule == 'alpha_linear':
        timesteps = np.arange(0, num_diffusion_timesteps+1, dtype=np.float64)/num_diffusion_timesteps
        alphas_cumprod = -timesteps+1
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = np.clip(betas, a_min=None, a_max=0.999)

    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.results_dir, "logs"),
    )
    if accelerator.is_main_process:

        if config.results_dir is not None:
            os.makedirs(config.results_dir, exist_ok=True)
        accelerator.init_trackers("train_example")


    csv_log_path = os.path.join(config.results_dir, "training_log.csv")
    
    if os.path.exists(csv_log_path):
        os.remove(csv_log_path)
    
    
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Step", "Loss", "Learning Rate", "Epoch Time(s)", "Cumulative Time(s)", "evaluation_mse"])

    model, optimizer, train_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, noise_scheduler
    )

    # beta_schedule: linear
    # beta_start: 0.0001
    # beta_end: 0.02
    # num_diffusion_timesteps: 1000

    betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000)

    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    # print("num_timesteps ", num_timesteps)

    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
    )
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    # num_timesteps =  len(train_dataloader) * config.num_epochs * config.train_batch_size
    scheduler_type = 'uniform'




    timesteps = 100




    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.


    csv_log_buffer = []

    global_step = 0
    cumulative_time = 0
    eval_mse = 0

    training_losses_list = []

    evaluation_results = []

    early_stopping = EarlyStopping(patience=500, min_delta=0, verbose=False)
    epoch_min_loss = 1000

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        epoch_start_time = time.time()
        epoch_loss = []

        eval_mse_epoch = []

        for step, (clean_images, cond_vectors) in enumerate(train_dataloader):
            # Sample noise to add to the images
            # noise = torch.randn(clean_images.shape)
            # clean_images = clean_images.to(device)
            # cond_vectors = cond_vectors.to(device)

            # noise = torch.randn_like(clean_images)
            # bs = clean_images.shape[0]

            # print("cond_vectors ", cond_vectors)
            # print("cond_vector train", cond_vectors, cond_vectors.shape)

            # Sample a random timestep for each image
            # timesteps = torch.randint(
            #     0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            # ).long()


            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            # noisy_images = noise_scheduler.add(clean_images, noise, timesteps)

            clean_images = clean_images.to(device)
            cond_vectors = cond_vectors.to(device)
            e = torch.randn_like(clean_images)
            b = betas

            if scheduler_type == 'uniform':
                skip = num_timesteps // timesteps
                t_intervals = torch.arange(-1, num_timesteps, skip)
                t_intervals[0] = 0
            elif scheduler_type == 'non-uniform':
                t_intervals = torch.tensor([0, 199, 399, 599, 699, 799, 849, 899, 949, 999])
                
                if timesteps != 10:
                    num_1 = int(timesteps*0.4)
                    num_2 = int(timesteps*0.6)
                    stage_1 = torch.linspace(0, 699, num_1+1)[:-1]
                    stage_2 = torch.linspace(699, 999, num_2)
                    stage_1 = torch.ceil(stage_1).long()
                    stage_2 = torch.ceil(stage_2).long()
                    t_intervals = torch.cat((stage_1, stage_2))
            else:
                raise Exception("The scheduler type is either uniform or non-uniform.")
                
            #  antithetic sampling
            idx_1 = torch.randint(0, len(t_intervals), size=(config.train_batch_size // 2 + 1,))
            idx_2 = len(t_intervals)-idx_1-1
            idx = torch.cat([idx_1, idx_2], dim=0)[:config.train_batch_size]
            t = t_intervals[idx].to(device)
            # print(t.size(), t)



            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                # noise_pred = model(noisy_images, timestep=timesteps, cond_vector=cond_vectors).sample

                # print(type(noise_pred), type(noise))
                # print(noise_pred.size(), noise.size())
                # loss = F.mse_loss(noise_pred, noise)

                # print(clean_images.device, t.device, e.device, b.device)
                loss = loss_fastddpm(model, clean_images, clean_images, t, e, b, cond_vectors)


                epoch_loss.append(loss.detach().item())

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                mean_epoch_loss = sum(epoch_loss)/len(epoch_loss)


                progress_bar.update(1)
                logs = {"loss": mean_epoch_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1



        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time += epoch_duration

        progress_bar.set_description(
        f"Epoch {epoch} | Time: {epoch_duration:.2f}s | Total: {cumulative_time:.2f}s")
        progress_bar.close()

        if eval_mse == 0:
            eval_mse = 0

        csv_log_buffer.append([epoch, global_step, mean_epoch_loss, lr_scheduler.get_last_lr()[0], epoch_duration, cumulative_time, eval_mse])


        pipeline = ConditionedDDPMPipeline(
                        unet=accelerator.unwrap_model(model),
                        scheduler=noise_scheduler,
                        image_size = config.image_size)
        


        training_losses_list.append(mean_epoch_loss)
        if mean_epoch_loss < epoch_min_loss:
            epoch_min_loss = mean_epoch_loss


            if (epoch+1) > 0:
                # torch.save(accelerator.unwrap_model(model).state_dict(), f"model_{config.results_dir}.pt")
                pipeline.save_pretrained(config.results_dir)

                checkpoint_path = os.path.join(config.results_dir, f"best_model.pt")
                torch.save({
                    "model": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "epoch_min_loss": epoch_min_loss,
                    "cumulative_time": cumulative_time,
                    "evaluation_mse": eval_mse
                }, checkpoint_path)


                if len(csv_log_buffer) > 0:
                    with open(csv_log_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_log_buffer)
                    csv_log_buffer.clear()

                    print("Saved ", epoch_min_loss)



        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:


            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:

                dfs = train_dataloader.dataset.df.sample(config.eval_batch_size)

                # Generate GT images for evaluation 
                gt_images = train_dataloader.dataset.generate_images_from_df(dfs)

                eval_mse_batch = evaluate(config, epoch, pipeline, dfs, gt_images)
                eval_mse = eval_mse_batch.mean()

                eval_mse_epoch = [(epoch + 1), eval_mse.item()]
                eval_mse_epoch.extend(eval_mse_batch.tolist())
                # print(eval_mse_epoch)

                # print(eval_mse_batch.tolist())
                # eval_mse_epoch.append(eval_mse_batch.tolist().insert(0,eval_mse.item()))
                # print(eval_mse_epoch)

                evaluation_results.append(eval_mse_epoch)
                # print(evaluation_results)

                evaluation_results_np = np.array(evaluation_results)
                # print("evaluation_results_np " , evaluation_results_np.shape)
                # print(evaluation_results_np)

                #Save training results to csv

                if len(csv_log_buffer) > 0:
                    with open(csv_log_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(csv_log_buffer)
                    csv_log_buffer.clear()

                    print("Saved ", epoch_min_loss)


                # Save evaluation results

                df_eval_results = pd.DataFrame(evaluation_results_np)
                df_eval_results.to_csv(f"{config.results_dir}/eval_mse.csv")

        early_stopping(mean_epoch_loss)
        if early_stopping.early_stop:
            print(f"Stopping training early at epoch {epoch}, loss: {mean_epoch_loss}")
            print(early_stopping.counter)
            print(early_stopping.best_loss)

            break


if __name__ == "__main__":
    config = TrainingConfig()
    # config.dataset_name = "huggan/smithsonian_butterflies_subset"
    # dataset = load_dataset(config.dataset_name, split="train")
    # dataset = load_dataset(config.dataset_name, cache_dir="/home/MichalMo/.cache/huggingface/datasets" , split="train")

    os.makedirs(config.results_dir, exist_ok=True)


    # print("PyTorch CUDA available:", torch.cuda.is_available())
    # print("Current device index:", torch.cuda.current_device() if torch.cuda.is_available() else "No GPU")
    # print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    # setup_cuda(use_memory_fraction=0.2, num_threads=4, device= "cpu")

    setup_cuda(use_memory_fraction=0.5, num_threads=8, visible_devices="0,1", use_cuda_with_id = 0)
    # setup_cuda(use_memory_fraction=0.9, num_threads=16, visible_devices="1,2",  multiGPU=True)
    
    device = torch.device("cuda", 0)
    # device = torch.device("cpu")
    
    
    # device0 = torch.device("cuda", 0)
    # device1 = torch.device("cuda", 1)
    # device2 = torch.device("cuda", 2)

    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print(device)


    dataset = ImageVectorDataset3DUSG(f"{config.dataset_dir}/images", f"{config.dataset_dir}/poses_unity.csv", image_size=config.image_size)
    dataset.preprocess()

    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, num_workers=16, shuffle=True)
    print(len(dataset))


    ##  stary model
    # base_unet = UNet2DConditionModel(
    #     sample_size=config.image_size, in_channels=1, out_channels=1, layers_per_block=2,
    #     block_out_channels=(64, 128, 256),  # now 4 blocks = 4 values
    #     down_block_types=('DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),

    #     up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D',  'UpBlock2D'),
    #     cross_attention_dim=128
    # )


    #######

    


    base_unet = UNet2DConditionModel(
        sample_size=config.image_size, in_channels=2, out_channels=1, layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),  # now 4 blocks = 4 values
        down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),

        up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
        cross_attention_dim=128
    )


    # model = nn.DataParallel(CustomConditionedUNet(base_unet, cond_dim=config.cond_dim, embedding_dim= config.cond_embedding_dim)).to(device)
    model = CustomConditionedUNet(base_unet, cond_dim=config.cond_dim, embedding_dim=config.cond_embedding_dim).to(device)
    # model = torch.nn.DataParallel(model).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=config.lr_warmup_steps,
    #     num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )



    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
        min_lr_ratio=0.25)  # e.g., ends at 10% of max LR

    



    # lr_scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=config.learning_rate,                  # Peak LR
    #         total_steps=(len(train_dataloader) * config.num_epochs),       # Total number of training steps
    #         pct_start=0.03,              # % of total steps used for warm-up
    #         anneal_strategy='cos',       # Cosine decay after warm-up
    #         cycle_momentum=False         # Disable momentum cycling for AdamW
    #     )



    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # print(dataset.labels_list)

    with open(f"{config.results_dir}/training_config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=4)

    # Save Dataset frame 
    dataset.df.to_csv(f"{config.results_dir}/dataset_df.csv")
    print(dataset.df.shape)


    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device)






