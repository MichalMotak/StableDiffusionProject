########################3

'''

Script for training DDPM for 3D Ultrasound dataset

dataset folder - dataset_Vol3, dataset_Vol3_2



'''

#################################3



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
import random 
import numpy as np


from PIL import Image
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
from torch import nn

import torchvision.transforms as T
from diffusers import UNet2DConditionModel
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
#######################

from test_gpu import setup_cuda






class CustomConditionedUNet(nn.Module):
    def __init__(self, base_unet: UNet2DConditionModel, cond_dim=6, embedding_dim=128):
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
    def __init__(self, unet, scheduler, image_size):
        super().__init__(unet=unet, scheduler=scheduler)
        self.image_size = image_size


    @torch.no_grad()
    def __call__(self, cond_vector, batch_size=1, generator=None, num_inference_steps=1000):

        sample_shape = (batch_size, self.unet.config.in_channels, self.image_size, self.image_size)  # adapt to your model
        print("sample_shape ", sample_shape)
        image = torch.randn(sample_shape, generator=generator).to("cuda")

        # print(cond_vector.size())

        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:

            model_output = self.unet(
                sample=image,
                timestep=t,
                cond_vector=cond_vector.to("cuda")  # pass conditioning vector
            )

            image = self.scheduler.step(model_output.sample, t, image).prev_sample

        return image





# class ImageVectorDataset(Dataset):
#     def __init__(self, image_dir, label_csv, image_size=64):
#         self.df = pd.read_csv(label_csv)
#         self.image_dir = image_dir
#         self.transform = T.Compose([
#             T.Resize((image_size, image_size)),
#             T.ToTensor(), 
#             T.Normalize([0.5], [0.5])
            
#         ])

#         self.labels_list = self.df['x'].values

#     def __len__(self):
#         return len(self.df)


#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         image = Image.open(os.path.join(self.image_dir, row['filename'])).convert("RGB")
#         # print("image ", image.size)
#         # image.show(title="before")
#         image = self.transform(image)

#         # print(torch.min(image), torch.max(image))

#         # image = image.clamp(0, 1) # -1 to 1
#         # back_to_pil = T.ToPILImage()(image)
#         # back_to_pil.show(title="after")

#         vector = torch.tensor([row['x'], row['y']], dtype=torch.float32)
#         return image, vector[:1]
    


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

        self.df = self.df[self.df['id'].isin(self.image_names)]
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
        rot_x_get = np.unique(self.df["rot_x"].values)[::2]
        rot_y_get = np.unique(self.df["rot_y"].values)
        rot_z_get = np.unique(self.df["rot_z"].values)[::2]

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

        image_path = os.path.join(self.image_dir, row['id'])
        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        vector = torch.tensor([row['pos_x'], row['pos_y'], row['pos_z'], row['rot_x'], row['rot_y'], row['rot_z']], dtype=torch.float32)
        return image, vector






@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 7000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 1
    save_image_epochs = 150
    save_model_epochs = 150
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results_3d_usvol3_1_128_again"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0




def my_make_grid(images, rows, cols):
    # w, h = images[0,0,:,:].size
    # w=64
    # h=64
    # grid = Image.new("RGB", size=(cols * w, rows * h))
    # for i, image in enumerate(images):
    #     grid.paste(image, box=(i % cols * w, i // cols * h))
    # return grid

    grid = make_grid(images, nrow=3, padding=2, normalize=True)

    # Convert to RGB PIL image
    grid_pil = to_pil_image(grid)
    return grid_pil



def evaluate(config, epoch, pipeline, df):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    # images = pipeline(
    #     batch_size=config.eval_batch_size,
    #     generator=torch.manual_seed(config.seed),
    # ).images
    # c = random.sample(list(labels_list), config.eval_batch_size)
    #print(torch.tensor(c, dtype = torch.long).unsqueeze(1).size())

    # print(df.head(5))
    dfs = df.sample(config.eval_batch_size)
    # print(dfs)
    c = torch.Tensor(dfs[['pos_x', 'pos_y', 'pos_z', "rot_x", "rot_y", "rot_z"]].values)
    # print(c, c.size())

        # vector = torch.tensor([row['pos_x'], row['pos_y'], row['pos_z'], row['rot_x'], row['rot_y'], row['rot_z']], dtype=torch.float32)


    images = pipeline(
                    cond_vector=torch.tensor(c, dtype = torch.half), # .unsqueeze(2)
                    batch_size=config.eval_batch_size)


    #print(images.shape)
    # Make a grid out of the images
    # image_grid = make_grid(images, rows=4, cols=4)
    image_grid = my_make_grid(images,rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")





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






def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    early_stopping = EarlyStopping(patience=500, min_delta=0, verbose=False)
    epoch_min_loss = 1000

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        epoch_loss = []

        for step, (clean_images, cond_vectors) in enumerate(train_dataloader):
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # print("cond_vectors ", cond_vectors)
            # print("cond_vector train", cond_vectors, cond_vectors.shape)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                noise_pred = model(noisy_images, timestep=timesteps, cond_vector=cond_vectors).sample

                loss = F.mse_loss(noise_pred, noise)
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


        pipeline = ConditionedDDPMPipeline(
                        unet=accelerator.unwrap_model(model),
                        scheduler=noise_scheduler,
                        image_size = config.image_size)
        


        if mean_epoch_loss < epoch_min_loss:
            epoch_min_loss = mean_epoch_loss


            if (epoch+1) > 100:
                torch.save(model.state_dict(), f"{config.output_dir}_cond_unet_vec_best.pt")
                pipeline.save_pretrained(config.output_dir)
                print("Saved ", epoch_min_loss)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            # pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            # pipeline.device = "cuda"
            # pipeline = pipeline.to("cuda")

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                df = train_dataloader.dataset.df

                evaluate(config, epoch, pipeline, df)

            # if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            #     if config.push_to_hub:
            #         repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
            #     else:
            #         pass
            #         # pipeline.save_pretrained(config.output_dir)
            #         #torch.save(model.state_dict(), f"{config.output_dir}_cond_unet_vec_{epoch + 1}.pt")


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


    # setup_cuda(use_memory_fraction=0.2, num_threads=4, visible_devices="0,1", use_cuda_with_id = 1)
    setup_cuda(use_memory_fraction=0.4, num_threads=8, visible_devices="0,1,2",  multiGPU=True)



    dataset = ImageVectorDataset3DUSG("dataset_Vol3_2/images", "dataset_Vol3_2/poses_unity.csv", image_size=config.image_size)
    dataset.preprocess()

    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    print(len(dataset))


    # df = train_dataloader.dataset.df
    # print(df.head(5))
    # dfs = df.sample(5)
    # print(dfs)
    # print(torch.Tensor(dfs[['x', 'y', 'z']].values))
    # print(torch.Tensor(dfs[['x', 'y', 'z']].values).size())


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
        sample_size=config.image_size, in_channels=1, out_channels=1, layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),  # now 4 blocks = 4 values
        down_block_types=('DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D',  'AttnDownBlock2D'),

        up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D',  'UpBlock2D', 'UpBlock2D'),
        cross_attention_dim=128
    )

    model = CustomConditionedUNet(base_unet, cond_dim=6, embedding_dim=128).cuda()
# 
    # print(base_unet.config)

    # print(model.config)
    # print(model.type)



    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )


# args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # print(dataset.labels_list)
    # print(random.sample(list(dataset.labels_list), 3))



    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)




