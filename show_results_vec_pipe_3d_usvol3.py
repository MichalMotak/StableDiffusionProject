
############3

"""

Script for evaluation for 3D Ultrasound dataset

dataset folder - dataset_Vol3, dataset_Vol3_2

"""

#############33


import json

from dataclasses import dataclass
from datasets import load_dataset
import matplotlib.pyplot as plt
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
from diffusers import DDPMPipeline
import math
import os
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from diffusers import UNet2DModel
from torchvision import transforms

from PIL import Image
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset

import torchvision.transforms as T

import numpy as np

from test_gpu import setup_cuda
from torch import nn
from diffusers import UNet2DConditionModel

import random
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


class CombinedImageVectorDataset3DUSG(Dataset):
    def __init__(self, dir_list, image_size=64):


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



        self.transform = T.Compose([
            # T.CenterCrop(300),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])



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
    def __init__(self, image_dir, label_csv, image_size=64):
        self.df = pd.read_csv(label_csv)
        self.image_dir = image_dir

        # Filter rows based on step from image filenames like image_0.png to image_359.png
        # self.df["index"] = self.df["filename"].str.extract(r"_(\d+)\.")[0].astype(int)
        # self.df = self.df[self.df["index"] % step == 0].reset_index(drop=True)

        # print(self.df)

        self.image_names = os.listdir(self.image_dir)
        print(len(self.image_names))

        self.df = self.df[self.df['filename'].isin(self.image_names)]
        print(self.df.shape)



        self.transform = T.Compose([
            T.CenterCrop(300),
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
        rot_x_get = np.unique(self.df["rot_x"].values)
        rot_y_get = np.unique(self.df["rot_y"].values)
        rot_z_get = np.unique(self.df["rot_z"].values)

        print(len(rot_x_get), rot_x_get)
        print(len(rot_y_get), rot_y_get)
        print(len(rot_z_get), rot_z_get)


        # for get in [rot_x_get, rot_y_get, rot_z_get]
        self.df = self.df[(self.df["rot_x"].isin(rot_x_get)) & (self.df["rot_y"].isin(rot_y_get)) & (self.df["rot_z"].isin(rot_z_get))]
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






class CustomConditionedUNet(nn.Module):
    def __init__(self, base_unet: UNet2DConditionModel, cond_dim=3, embedding_dim=128):
        super().__init__()
        self.unet = base_unet
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.config = self.unet.config



    def forward(self, sample, timestep, cond_vector=None):
        # [B, 128] -> [B, 1, 128] (cross-attention format)
        cond_emb = self.cond_embed(cond_vector).unsqueeze(1)
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=cond_emb
        )



class ConditionedDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, cond_vector, batch_size=1, generator=None, num_inference_steps=1000):

        sample_shape = (batch_size, self.unet.config.in_channels, 128, 128)  # adapt to your model
        image = torch.randn(sample_shape, generator=generator).to("cuda")

        #print(cond_vector.size())

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            model_output = self.unet(
                sample=image,
                timestep=t,
                cond_vector=cond_vector.to("cuda")  # pass conditioning vector
            )
            image = self.scheduler.step(model_output.sample, t, image).prev_sample

        return image



def my_make_grid(images, rows, cols):
    # w, h = images[0,0,:,:].size
    # w=64
    # h=64
    # grid = Image.new("RGB", size=(cols * w, rows * h))
    # for i, image in enumerate(images):
    #     grid.paste(image, box=(i % cols * w, i // cols * h))
    # return grid

    grid = make_grid(images, nrow=4, padding=2, normalize=True)

    # Convert to RGB PIL image
    grid_pil = to_pil_image(grid)
    return grid_pil



def evaluate(pipeline, labels_list):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    #c = random.sample(list(labels_list), 16)


    print(labels_list, labels_list.size())

    images = pipeline(
                    cond_vector=torch.tensor(labels_list, dtype = torch.float),
                    batch_size=labels_list.size()[0],
                    generator=torch.manual_seed(0)
                    )



    print(images.shape)
    print(torch.min(images[0]), torch.max(images[0]))
    print(torch.min((images[0]+1)/2), torch.max((images[0]+1)/2))

    # Make a grid out of the images
    image_grid = my_make_grid(images,rows=4, cols=4)
    image_grid.show()

    #Save the images
    # os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{SAVE_SHOW_RESULTS_PATH}/grid.png")

    # save images 

    for i, c2 in enumerate(labels_list):
        n = c2.cpu().detach().numpy()
        print(n)
        image = to_pil_image(((images[i]+1)/2))
        print(np.asarray(image).max(), np.asarray(image).min())
        image.save(f"{SAVE_SHOW_RESULTS_PATH}/image_{i}_{n[0]:.3f}_{n[1]:.3f}_{n[2]:.3f}_{n[3]:.3f}_{n[4]:.3f}_{n[5]:.3f}.png")







# setup_cuda(use_memory_fraction=0.1, num_threads=2, visible_devices="0,1", use_cuda_with_id = 0)


# base_unet = UNet2DConditionModel(
#     sample_size=64, in_channels=1, out_channels=1, layers_per_block=2,
#     block_out_channels=(64, 128, 256),  # now 4 blocks = 4 values
#     down_block_types=('DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),
#     up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D',  'UpBlock2D'),
#     cross_attention_dim=128
# )


setup_cuda(use_memory_fraction=0.2, num_threads=4, visible_devices="0,1", use_cuda_with_id = 0)


# Model was not saved in checkpoint

# base_unet = UNet2DConditionModel(
#     sample_size=128, in_channels=1, out_channels=1, layers_per_block=2,
#     block_out_channels=(64, 128, 256, 512),  # now 4 blocks = 4 values
#     down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),

#     up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D',  'UpBlock2D', 'UpBlock2D'),
#     cross_attention_dim=128
# )


# base_unet = UNet2DConditionModel(
#     sample_size=128, in_channels=1, out_channels=1, layers_per_block=2,
#     block_out_channels=(64, 128, 256, 512),  # now 4 blocks = 4 values
#     down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),

#     up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D','UpBlock2D', 'UpBlock2D'),
#     cross_attention_dim=128
# )



# model = CustomConditionedUNet(base_unet, cond_dim = 6).cuda()


# # model.load_state_dict(torch.load(f"model_{RESULTS_PATH}.pt", weights_only=True))

# RESULTS_PATH = ""
# SAVE_SHOW_RESULTS_PATH = F"show_{RESULTS_PATH}"



# # Load model from checkpoint
# resume_from_checkpoint = f"{RESULTS_PATH}/best_model.pt"


# if not os.path.exists(SAVE_SHOW_RESULTS_PATH):
#     os.makedirs(SAVE_SHOW_RESULTS_PATH)


# with open(f'{RESULTS_PATH}/training_config.json') as f:
#     training_config = json.load(f)
#     print(training_config)
#     print(training_config['dataset_dir'])


# # pipeline = ConditionedDDPMPipeline.from_pretrained(training_config['results_dir'])
# # model = pipeline.unet

# checkpoint = torch.load(resume_from_checkpoint, map_location="cuda", weights_only=False)
# # model.load_state_dict(torch.load(f"{RESULTS_PATH}/best_model.pt", weights_only=True))


# # checkpoint = torch.load(resume_from_checkpoint, map_location="cuda")
# model.load_state_dict(checkpoint["model"])



# dataset = ImageVectorDataset3DUSG(f"{training_config['dataset_dir']}/images", f"{training_config['dataset_dir']}/poses_unity.csv", 
#                                   image_size=training_config['image_size'])
# # dataset.preprocess()


# train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# print(len(dataset))


# dfs = dataset.df.sample(4, random_state=1)
# c = torch.Tensor(dfs[training_config['cond_parameters']].values)

# print(c.size())


# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# pipeline = ConditionedDDPMPipeline(
#                 unet=model,
#                 scheduler=noise_scheduler)





# evaluate(pipeline, c)





# STARY KOD DLA STARYCH MODELI


#
# 



# base_unet = UNet2DConditionModel(
#     sample_size=128, in_channels=1, out_channels=1, layers_per_block=2,
#     block_out_channels=(64, 128, 256, 256, 512),  # now 4 blocks = 4 values
#     down_block_types=('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),

#     up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
#     cross_attention_dim=128
# )


base_unet = UNet2DConditionModel(
    sample_size=128, in_channels=1, out_channels=1, layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),  # now 4 blocks = 4 values
    down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),

    up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
    cross_attention_dim=128
)


model = CustomConditionedUNet(base_unet, cond_dim = 6).cuda()

RESULTS_PATH = "show_results_dataset_Vol3_345_f_2_cont"
resume_from_checkpoint = f"{RESULTS_PATH}/best_model.pt"


# checkpoint = torch.load(resume_from_checkpoint, map_location="cuda")
model.load_state_dict(torch.load(f"{RESULTS_PATH}/best_model.pt", weights_only=False)['model'])


# checkpoint = torch.load(resume_from_checkpoint, map_location="cuda")
# model.load_state_dict(checkpoint["model"])



SAVE_SHOW_RESULTS_PATH = F"show_{RESULTS_PATH}"

if not os.path.exists(SAVE_SHOW_RESULTS_PATH):
    os.makedirs(SAVE_SHOW_RESULTS_PATH)


dataset_csv_path = "dataset_cropped3_f/poses_unity.csv"
original_images_dir = "dataset_cropped3_f/images"

df = pd.read_csv(dataset_csv_path)

img_list = os.listdir(original_images_dir)
# img_list


mask = df['filename'].isin(img_list)
df = df[mask]




dataset = CombinedImageVectorDataset3DUSG(["dataset_Vol3_f", "dataset_Vol4_f", "dataset_Vol5_f"], image_size=128)

df = dataset.df

dfs = df.sample(4, random_state=128)
# c = torch.Tensor(dfs[training_config['cond_parameters']].values)


# c =  torch. tensor(df. values[1:, :])

c =  torch. tensor(dfs.values[:, 1:].astype(np.float32))
print(c)
# c = torch.Tensor(dfs[training_config['cond_parameters']].values)

print(c.size())


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


pipeline = ConditionedDDPMPipeline(
                unet=model,
                scheduler=noise_scheduler)



# model.load_state_dict(torch.load(f"model_{RESULTS_PATH}.pt", weights_only=True))
# 


evaluate(pipeline, c)









