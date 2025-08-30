
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
from torch import device, nn
from torch.optim.lr_scheduler import OneCycleLR

from torch.optim.lr_scheduler import LambdaLR

import torchvision.transforms as T
from diffusers import UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
#######################

from test_gpu import setup_cuda

accelerator = Accelerator()


@dataclass
class TrainingConfig:

    image_size: int = 128  # the generated image resolution
    image_format:str = "gray"

    train_batch_size: int  = 64
    eval_batch_size: int  = 32  # how many images to sample during evaluation

    num_epochs: int = 1000
    save_image_epochs: int  = 50
    save_model_epochs: int  = 50

    
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4 * 0.35
    lr_warmup_steps: int = 0

    mixed_precision: str = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    results_dir: str = "results_dataset_Vol3_f_2_cont" 
    dataset_dir: str = "dataset_Vol3_f"


    cond_dim: int = 6
    cond_embedding_dim: int = 128
    cond_parameters: List = field(default_factory=lambda: ['pos_x', 'pos_y', 'pos_z', "rot_x", "rot_y", "rot_z"])

    multi_gpu: bool = True

    def get_transforms(self, input = None):
        ops = []

        if input:
            for t in input:
                ops.append(eval(f"T.{t}"))

        else:
            for t in self.transform:
                ops.append(eval(f"T.{t}"))


        return T.Compose(ops)


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




class CombinedImageVectorDataset3DUSG(Dataset):
    def __init__(self, dir_list, image_size, transforms):


        self.image_names_list = []
        self.image_names_only_list = []
    
        self.label_csv_list = []

        for dir in dir_list:
            
            label_csv = os.path.join(dir, "poses_unity.csv")
            image_dir = os.path.join(dir, "images")

            image_names = os.listdir(image_dir)

            df = pd.read_csv(label_csv)
            print(df.shape)
            print(df.head(3))
            print("image_names ", len(image_names))
            # self.image_dir = image_dir
            # df = df[df['filename'].isin(image_names)]
            print(df.shape)

            self.label_csv_list.append(df)

            self.image_names_only_list.extend(image_names)



            for img_name in image_names:

                self.image_names_list.append(os.path.join(image_dir,img_name))


            print(len(self.image_names_list))
            print(len(self.image_names_only_list))

        print(self.image_names_list[:5])
        print(self.image_names_only_list[:5])


        self.df = pd.concat(self.label_csv_list, ignore_index = True)

        self.t = 0
        print(self.df.shape)



        self.transform = transforms

        print("dataset transform", self.transform)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # print(idx)




        # print(row)
        # print(row.values)
        # print(row.values[0][1:])
        # print(type(row.values[0][1:]))
        # print(row["pos_x"].values)

        
        # image = Image.open(image_path).convert("L")
        # image = self.transform(image)

        # # vector = torch.tensor([row['pos_x'].values, row['pos_y'].values, row['pos_z'].values, 
        # #                        row['rot_x'].values, row['rot_y'].values, row['rot_z'].values], dtype=torch.float32)
    
        # vector = torch.from_numpy(row.values[0][1:].astype(np.float32))
        # return image, vector
    

        try:
            image_path = self.image_names_list[idx]
        # print(image_path)

            row = self.df[self.df["filename"] == image_path.split("/")[-1]]
            image = Image.open(image_path).convert("L")
            
            image = self.transform(image)
            self.lastimg = image

            
        except Exception as e:
            self.t+=1

            image = self.lastimg

            print(idx)
                        
            print(image_path)
            print(row)
            
        vector = torch.from_numpy(row.values[0][1:].astype(np.float32))
        return image, vector #image_path.split("/")[-1]




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

class ImageVectorDataset3DUSG(Dataset):
    def __init__(self, image_dir, label_csv, image_size, transforms):
        self.df = pd.read_csv(label_csv)
        self.image_dir = image_dir

        # Filter rows based on step from image filenames like image_0.png to image_359.png
        # self.df["index"] = self.df["filename"].str.extract(r"_(\d+)\.")[0].astype(int)
        # self.df = self.df[self.df["index"] % step == 0].reset_index(drop=True)

        # print(self.df)

        self.image_names = os.listdir(self.image_dir)
        print("image_names ", len(self.image_names))

        self.df = self.df[self.df['filename'].isin(self.image_names)]
        print(self.df.shape)


        self.transform = transforms

        print("dataset transform", self.transform)


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



def interleave_tensors(images, gt_images):

    # images = images.permute(1,0,2,3 )
    gt_images = gt_images.to(images.device)
    gt_images = gt_images.permute(1,0,2,3)

    B1 = images.shape[0]
    B2 = gt_images.shape[0]
    print(images.shape, gt_images.shape)

    C, H, W = images.shape[1], images.shape[2], images.shape[3]

    # Find the smaller and larger batch size
    min_len = min(B1, B2)
    max_len = max(B1, B2)

    # Interleave first min_len elements
    # print([images[:,:min_len].shape, gt_images[:,:min_len]].shape)
    interleaved = torch.stack([images[:min_len], gt_images[:min_len]], dim=1)
    print(interleaved.shape)
    interleaved = interleaved.reshape(-1, C, H, W)
    print(interleaved.shape)

    # Append remaining items from the longer tensor

    remainder = gt_images[min_len:]

    # Concatenate
    result = torch.cat([interleaved, remainder], dim=0)
    return result


def my_make_grid(images, gt_images, rows):

    images_to_grid = interleave_tensors(images, gt_images)
    print("images_to_grid ", images_to_grid.shape)

    grid = make_grid(images_to_grid, nrow=rows, padding=5, normalize=True)

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
    


    print(images.shape, gt_images.shape)
    # print(images.device, gt_images[0].device)
    error = mse(images[:,0,:,:], gt_images[0].to(images.device))
    # print(error.size())
    
    # print(error.mean())


    image_grid = my_make_grid(images, gt_images, rows=2)

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






def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, resume_from_checkpoint):


    # global_step = 438001
    # cumulative_time = 86323.49588918686
    # epoch_min_loss = 0.0020921836977451



    # === Load checkpoint if resuming ===
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        print(f"🔁 Loading checkpoint from {resume_from_checkpoint}")

        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["epoch"]
        # epoch_min_loss = checkpoint.get("epoch_min_loss", epoch_min_loss)
        # cumulative_time = checkpoint.get("cumulative_time", cumulative_time)
        epoch_min_loss = checkpoint['epoch_min_loss']
        cumulative_time = df_training_results.iloc[-1]['Cumulative Time(s)']

        print(lr_scheduler)

        print(optimizer)

        print(global_step, start_epoch)
        print(epoch_min_loss, cumulative_time)

        print(cumulative_time)

    # Initialize accelerator and tensorboard logging

    if accelerator.is_main_process:

        if config.results_dir is not None:
            os.makedirs(config.results_dir, exist_ok=True)
        accelerator.init_trackers("train_example")



    csv_log_path = os.path.join(config.results_dir, "training_log.csv")
    
    if os.path.exists(csv_log_path):
        os.remove(csv_log_path)
    
    with open(csv_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Step", "Loss", "Learning Rate", "Epoch Time(s)", "Cumulative Time(s)"])






    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, noise_scheduler
    )


    csv_log_buffer = []
    eval_mse = 0
    training_losses_list = []
    evaluation_results = []
    early_stopping = EarlyStopping(patience=500, min_delta=0, verbose=False)
    epoch_min_loss = 1000

    for epoch in range(config.num_epochs):

        # print(f"[Process {accelerator.process_index}] running on {accelerator.device}")

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        epoch_start_time = time.time()
        epoch_loss = []

        for step, (clean_images, cond_vectors) in enumerate(train_dataloader):
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timestep=timesteps, cond_vector=cond_vectors).sample
                loss = F.mse_loss(noise_pred, noise)
                epoch_loss.append(loss.detach().item())
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)
                progress_bar.update(1)
                logs = {"loss": mean_epoch_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cumulative_time += epoch_duration
        progress_bar.set_description(f"Epoch {epoch} | Time: {epoch_duration:.2f}s | Total: {cumulative_time:.2f}s")
        progress_bar.close()

        if eval_mse == 0:
            eval_mse = 0

        csv_log_buffer.append([epoch, global_step, mean_epoch_loss, lr_scheduler.get_last_lr()[0], epoch_duration, cumulative_time, eval_mse])

        if accelerator.is_main_process:
            pipeline = ConditionedDDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
                image_size=config.image_size
            )
            training_losses_list.append(mean_epoch_loss)

            if mean_epoch_loss < epoch_min_loss:
                epoch_min_loss = mean_epoch_loss
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
                with open(os.path.join(config.results_dir, "training_log.csv"), mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_log_buffer)
                csv_log_buffer.clear()

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                dfs = train_dataloader.dataset.df.sample(config.eval_batch_size)
                gt_images = train_dataloader.dataset.generate_images_from_df(dfs)
                eval_mse_batch = evaluate(config, epoch, pipeline, dfs, gt_images)
                eval_mse = eval_mse_batch.mean()
                eval_mse_epoch = [(epoch + 1), eval_mse.item()]
                eval_mse_epoch.extend(eval_mse_batch.tolist())
                evaluation_results.append(eval_mse_epoch)
                df_eval_results = pd.DataFrame(np.array(evaluation_results))
                df_eval_results.to_csv(f"{config.results_dir}/eval_mse.csv")

        early_stopping(mean_epoch_loss)
        if early_stopping.early_stop:
            print(f"Stopping training early at epoch {epoch}, loss: {mean_epoch_loss}")
            break



if __name__ == "__main__":
    config = TrainingConfig()

    os.makedirs(config.results_dir, exist_ok=True)


    if config.multi_gpu:
        setup_cuda(use_memory_fraction=0.8, num_threads=16, visible_devices="0,1,2", multiGPU=True)
    else:
        setup_cuda(use_memory_fraction=0.8, num_threads=16, visible_devices="0,1", use_cuda_with_id = 0)



    RESULTS_PATH = "results_dataset_Vol3_f_2"

    with open(f'{RESULTS_PATH}/training_config.json') as f:
        training_config = json.load(f)
        print(training_config)
        print(training_config['dataset_dir'])

    
    df_training_results = pd.read_csv(f'{RESULTS_PATH}/training_log.csv')

    # device = torch.device("cuda", 2)

    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print(device)


    # dataset = ImageVectorDataset3DUSG(f"{training_config['dataset_dir']}/images", f"{training_config['dataset_dir']}/poses_unity.csv", image_size=config.image_size)
    # dataset.preprocess()

    dataset_transform = config.get_transforms(training_config['transform'])

    # dataset = CombinedImageVectorDataset3DUSG(["dataset_Vol3_f", "dataset_Vol4_f", "dataset_Vol5_f"], image_size=training_config['image_size'], transforms = dataset_transform)
    dataset = ImageVectorDataset3DUSG(f"{training_config['dataset_dir']}/images", f"{training_config['dataset_dir']}/poses_unity.csv", image_size=training_config['image_size'], transforms = dataset_transform)


    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, num_workers=16, shuffle=True) #training_config['train_batch_size']
    print("dataset len", len(dataset))
    print("dataloader len ", len(train_dataloader))


    pipeline = ConditionedDDPMPipeline.from_pretrained(training_config['results_dir'])
    model = pipeline.unet


    # print(pipeline)
    # print(model)

    # print(pipeline.scheduler)
    # print(pipeline)


    resume_from_checkpoint = f"{RESULTS_PATH}/best_model.pt"

    # checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")


    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # optimizer.lr = 123

    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=1,
    #     num_training_steps=(len(train_dataloader) * config.num_epochs),
    # )


    # lr_scheduler = OneCycleLR(
    #         optimizer,
    #         max_lr=training_config["learning_rate"],                  # Peak LR
    #         total_steps=(len(train_dataloader) * config.num_epochs),       # Total number of training steps
    #         pct_start=0.03,              # % of total steps used for warm-up
    #         anneal_strategy='cos',       # Cosine decay after warm-up
    #         cycle_momentum=False         # Disable momentum cycling for AdamW
    #     )


    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
        min_lr_ratio=0.001)  # e.g., ends at 10% of max LR


    # print(checkpoint["lr_scheduler"])

    print(lr_scheduler)

    print(optimizer)


    # lr_sch_check =checkpoint["lr_scheduler"]
    # lr_sch_check['_last_lr'] = 0.123

    # lr_sch_opt =checkpoint["optimizer"]
    # lr_sch_opt['lr'] = 0.0000005

    # model.load_state_dict(checkpoint["model"])
    # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    # optimizer.load_state_dict(checkpoint["optimizer"])


    # global_step = checkpoint["global_step"]
    # start_epoch = checkpoint["epoch"]


    # noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler = pipeline.scheduler




    # print(dataset.labels_list)

    # with open(f"{config.results_dir}/training_config.json", "w") as f:
    #     json.dump(dataclasses.asdict(config), f, indent=4)

    # # Save Dataset frame 
    # dataset.df.to_csv(f"{config.results_dir}/dataset_df.csv")
    # print(dataset.df.shape)




    # print(f"🔁 Loading checkpoint from {resume_from_checkpoint}")

    # checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")

    # model.load_state_dict(checkpoint["model"])
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    # global_step = checkpoint["global_step"]
    # start_epoch = checkpoint["epoch"]
    # # epoch_min_loss = checkpoint.get("epoch_min_loss", epoch_min_loss)
    # # cumulative_time = checkpoint.get("cumulative_time", cumulative_time)
    # epoch_min_loss = checkpoint['epoch_min_loss']
    # cumulative_time = df_training_results.iloc[-1]['Cumulative Time(s)']

    # print(lr_scheduler)

    # print(optimizer)

    # print(global_step, start_epoch)
    # print(epoch_min_loss, cumulative_time)

    # print(cumulative_time)


    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, resume_from_checkpoint)




