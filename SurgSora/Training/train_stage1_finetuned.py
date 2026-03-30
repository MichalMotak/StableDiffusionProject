#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from segment_anything import sam_model_registry, SamPredictor

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import logging
import math
import time
import csv
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
import imageio
import accelerate
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import pickle
import torch
import torchvision
import torch.nn.functional as F
import lpips

import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

# from train_utils.dataset import WebVid10M, ESDDataset
import models as mod
from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import DualFlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import DualFlowControlNet_traj

from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_utils.unimatch.utils.flow_viz import flow_to_image
from train_utils.unimatch.utils.visualization import viz_depth_tensor

from train_utils.dav2.depth_anything_v2.dpt import DepthAnythingV2
# from segment_anything import sam_model_registry, SamPredictor
import matplotlib
import cv2
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as FT
from test_gpu import setup_cuda

from safetensors.torch import load_file
from kohya_lora2 import LoRANetwork

# import tempfile
# tempfile.tempdir = './DUMMY_PATH'

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")





def warp_with_mask(x, flow):
    """
    x: [B, C, H, W]
    flow: [B, 2, H, W]
    """
    B, C, H, W = x.size()
    # Create identity grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Add flow and normalize
    vgrid = grid + flow.permute(0, 2, 3, 1)
    vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    
    # Warp the image
    output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    # Warp a mask of ones to find 'invalid' out-of-bounds pixels
    mask = torch.ones((B, 1, H, W), device=x.device)
    valid_mask = F.grid_sample(mask, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    valid_mask = (valid_mask > 0.99).float() # Threshold to remove interpolation artifacts at edges
    
    return output, valid_mask



def get_consecutive_optical_flows(unimatch, video_frame):
    '''
        video_frame: [b, t, c, w, h]
    '''

    video_frame = video_frame * 255

    # print(video_frame.dtype)

    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, i], video_frame[:, i + 1]
        # print(image1.dtype)
        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
        # print(image1_r.dtype)
        results_dict_r = unimatch(image1_r, image2_r,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
            )
        flow_r = results_dict_r['flow_preds'][-1]  # [b, 2, H, W]
        # print(flow_r.shape)
        flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
        flows.append(flow.unsqueeze(1))  # [b, 1, 2, w, h]
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t, 2, w, h]
    return flows


def apply_semantic_mask_to_flow(flow, masks):
    """
    flow: [b, t, 2, h, w]
    masks: [b, t, 1, h, w] - Binary (0 for background, 1 for tools/eye)
    """
    # print("flow ", flow.shape, masks.shape)
    # Ensure masks are boolean or float 0/1
    # We expand the mask to cover both U and V channels
    masks[masks>0] = 1

    masked_flow = flow * masks.repeat((1, 1, 2, 1, 1))
    return masked_flow

def apply_masked_blur_to_flow(flow, masks, kernel_size=31, sigma=10.0):
    """
    flow: [b, t, 2, h, w] - Raw optical flow
    masks: [b, t, 1, h, w] - Binary mask (1 for tools, 0 for background)
    kernel_size: Size of the blur. Larger = smoother background.
    """
    b, t, c, h, w = flow.shape

    masks[masks>0] = 1

    
    # 1. Reshape to [Batch*Time, Channels, H, W] for fast batch processing
    flow_flat = flow.view(-1, c, h, w)
    
    # 2. Generate the blurred version of the entire flow field
    # We use a large kernel to remove the "pixel-noise" seen in real footage
    blurred_flow_flat = FT.gaussian_blur(flow_flat, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    blurred_flow = blurred_flow_flat.view(b, t, c, h, w)
    
    # 3. Expand mask to cover both U and V flow channels
    mask_expanded = masks.repeat(1, 1, 2, 1, 1) # [b, t, 2, h, w]
    inverse_mask = 1 - mask_expanded
    
    # 4. Combine: Sharp tools + Blurred background
    # (flow * mask) -> Keeps tool motion exact
    # (blurred_flow * inverse_mask) -> Smoothes the eye/iris motion
    processed_flow = (flow * mask_expanded) + (blurred_flow * inverse_mask)
    
    return processed_flow




def overlay_segmentation(image, rgb_mask, alpha=0.5):
    """
    Overlay a color segmentation mask on top of an image with transparency.

    Args:
        image (np.ndarray): Original image, shape (H, W, 3), dtype uint8, range [0,255].
        rgb_mask (np.ndarray): Segmentation mask in RGB colors, same shape and dtype.
        alpha (float): Transparency for the mask. 0 = only image, 1 = only mask.

    Returns:
        np.ndarray: Blended RGB image, dtype uint8, shape (H, W, 3).
    """
    assert image.shape == rgb_mask.shape, f"Shape mismatch: {image.shape} vs {rgb_mask.shape}"
    assert 0.0 <= alpha <= 1.0, "alpha must be between 0 and 1"

    # Convert to float for blending
    image_f = image.astype(np.float32)
    mask_f = rgb_mask.astype(np.float32)

    # Blend: weighted average
    blended = (1 - alpha) * image_f + alpha * mask_f

    # Clip and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def grayscale_to_color_mask(seg_image, dataset):
    """
    Convert grayscale-like segmentation image [H, W, 3] with values 0–13
    into a color RGB visualization [H, W, 3].

    Args:
        seg_image (np.ndarray): Input segmentation image, shape (H, W, 3), dtype uint8,
                                values in range [0, 13].
    Returns:
        np.ndarray: RGB color image of shape (H, W, 3), dtype uint8.
    """
    # assert seg_image.ndim == 3 and seg_image.shape[2] == 3, \
    #     f"Expected [H,W,3], got {seg_image.shape}"

    # Convert to single-channel class map
    class_map = seg_image.astype(np.uint8)  # values 0–13

    # Define a color palette (you can customize)
    if dataset =="cataract":
        color_palette = np.array([
            [  0,   0,   0],   # 0: background
            [128,   0,   0],   # 1
            [  0, 128,   0],   # 2
            [128, 128,   0],   # 3
            [  0,   0, 128],   # 4
            [128,   0, 128],   # 5
            [  0, 128, 128],   # 6
            [128, 128, 128],   # 7
            [ 64,   0,   0],   # 8
            [192,   0,   0],   # 9
            [ 64, 128,   0],   # 10
            [192, 128,   0],   # 11
            [ 64,   0, 128],   # 12
            [192,   0, 128],   # 13
        ], dtype=np.uint8)

    elif dataset =="cadis":
        color_palette = np.array([
        [  0,   0,   0],   # 0  background
        [128,   0,   0],   # 1
        [  0, 128,   0],   # 2
        [128, 128,   0],   # 3
        [  0,   0, 128],   # 4
        [128,   0, 128],   # 5
        [  0, 128, 128],   # 6
        [128, 128, 128],   # 7
        [ 64,   0,   0],   # 8
        [192,   0,   0],   # 9
        [ 64, 128,   0],   # 10
        [192, 128,   0],   # 11
        [ 64,   0, 128],   # 12
        [192,   0, 128],   # 13
        [ 64, 128, 128],   # 14
        [192, 128, 128],   # 15
        [  0,  64,   0],   # 16
        [128,  64,   0],   # 17
        [  0, 192,   0],   # 18
        [128, 192,   0],   # 19
        [  0,  64, 128],   # 20
        [128,  64, 128],   # 21
        [  0, 192, 128],   # 22
        [128, 192, 128],   # 23
        [ 64,  64,   0],   # 24
        [192,  64,   0],   # 25
        [ 64, 192,   0],   # 26
        [192, 192,   0],   # 27
        [ 64,  64, 128],   # 28
        [192,  64, 128],   # 29
        [ 64, 192, 128],   # 30
        [192, 192, 128],   # 31
        [  0,  64, 192],   # 32
        [128,  64, 192],   # 33
        [  0, 192, 192],   # 34
        [255, 255, 255],   # 35  (white / special)
    ], dtype=np.uint8)

    # Map class indices to colors
    rgb_mask = color_palette[class_map]  # shape [H, W, 3]

    return rgb_mask


class DatasetMultimodalVideoSimToReal(Dataset):
    def __init__(self, image_size = 256,  main_path=None, record = None, num_frames = 21, limit_frames = None):
        self.data = []


        self.num_frames = num_frames
        self.record = record
        self.main_path = main_path
        case_list = sorted(os.listdir(os.path.join(self.main_path, record)))
        self.image_numbers = [x[-8:-4] for x in case_list]

        x = len(self.image_numbers)
        self.image_numbers = self.image_numbers[:x//2]

        if limit_frames:
            assert (limit_frames >= num_frames)
            self.image_numbers = self.image_numbers[:limit_frames]

        self.image_shape = (image_size, image_size)
        # print(len(self.image_numbers))
        # print(self.image_numbers)

        print(f"Loaded {len(self.image_numbers)} samples")


    def __len__(self):
        return len(self.image_numbers) 



    def load_image(self, num, img_type):

        assert img_type in ["image", "mask", "depth"]



        if img_type == "image":
            path = os.path.join(self.main_path, self.record, f"frame_{num}.png")

        elif img_type == "mask":
            path = os.path.join(self.main_path, self.record+ "_final", f"mask_{num}.png")

        elif img_type =="depth":
            path = os.path.join(self.main_path, self.record+ "_depth_synth", f"depth_{num}.png")

        # print(path)
        image = Image.open(path)

        image = np.asarray(image)

        if img_type == "image":

            image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)
            image = np.moveaxis(image, -1, 0)
            image = image / 255.

        elif img_type == "depth":
            image = np.moveaxis(image, -1, 0)


        return image

    def get_batch(self, indexes):

        # base_idx = idx % len(self.df)
        image_list = []
        mask_list = []
        depth_list = []


        for idx in indexes:
            num = self.image_numbers[idx]

            # print(num)

            image = self.load_image(num, "image")
            mask = self.load_image(num, "mask")
            depth = self.load_image(num, "depth")

            image_list.append(image)
            mask_list.append(mask)
            depth_list.append(depth)
        

        images = np.stack(image_list)
        masks = np.stack(mask_list)
        depths = np.stack(depth_list)

        return images, masks, depths


    def __getitem__(self, idx):


        max_start = len(self.image_numbers) - self.num_frames

        # one random iteration
        start_idx = random.randint(0, max_start)
        batch_indexes = range(start_idx, start_idx + self.num_frames)


        images, masks, depths = self.get_batch(batch_indexes)

        # print(images.shape, masks.shape, depths.shape)
        video_name = f"video_{self.record}_{start_idx}_{start_idx + self.num_frames}"

        image_diff_path = f"/home/MichalMo/projects/ControlNet-diffusers/records_cadis_1/diff_results/{self.record}/diff_output_{self.image_numbers[start_idx]}.png"

        image_diff = Image.open(image_diff_path)
        image_diff = np.asarray(image_diff)

        image_diff = cv2.resize(image_diff, self.image_shape, interpolation=cv2.INTER_NEAREST)
        image_diff = np.moveaxis(image_diff, -1, 0)
        image_diff = image_diff / 255.

        images[0] = image_diff

        # print("prompt", prompt, type(prompt)) # depth = torch.tensor(depth)
        return dict(image=torch.tensor(images), mask=torch.tensor(masks), depth = torch.tensor(depths),  video_name =video_name)






class DatasetMultimodalVideo(Dataset):
    def __init__(self, image_size = 256,  main_path=None, record = None, num_frames = 21, limit_frames = None):
        self.data = []


        self.num_frames = num_frames
        self.record = record
        self.main_path = main_path
        case_list = sorted(os.listdir(os.path.join(self.main_path, record)))
        self.image_numbers = [x[-8:-4] for x in case_list]

        if limit_frames:
            assert (limit_frames >= num_frames)
            self.image_numbers = self.image_numbers[:limit_frames]

        self.image_shape = (image_size, image_size)
        # print(len(self.image_numbers))
        # print(self.image_numbers)

        print(f"Loaded {len(self.image_numbers)} samples")


    def __len__(self):
        return len(self.image_numbers) 



    def load_image(self, num, img_type):

        assert img_type in ["image", "mask", "depth"]



        if img_type == "image":
            path = os.path.join(self.main_path, self.record, f"frame_{num}.png")

        elif img_type == "mask":
            path = os.path.join(self.main_path, self.record+ "_final", f"mask_{num}.png")

        elif img_type =="depth":
            path = os.path.join(self.main_path, self.record+ "_depth_synth", f"depth_{num}.png")


        image = Image.open(path)

        image = np.asarray(image)

        if img_type == "image":

            image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)
            image = np.moveaxis(image, -1, 0)
            image = image / 255.

        elif img_type == "depth":
            image = np.moveaxis(image, -1, 0)


        return image



    def get_batch(self, indexes):

        # base_idx = idx % len(self.df)
        image_list = []
        mask_list = []
        depth_list = []


        for idx in indexes:
            num = self.image_numbers[idx]

            # print(num)

            image = self.load_image(num, "image")
            mask = self.load_image(num, "mask")
            depth = self.load_image(num, "depth")

            image_list.append(image)
            mask_list.append(mask)
            depth_list.append(depth)
        

        images = np.stack(image_list)
        masks = np.stack(mask_list)
        depths = np.stack(depth_list)

        return images, masks, depths


    def __getitem__(self, idx):


        max_start = len(self.image_numbers) - self.num_frames

        # one random iteration
        start_idx = random.randint(0, max_start)
        batch_indexes = range(start_idx, start_idx + self.num_frames)
        # print(batch_indexes)


        images, masks, depths = self.get_batch(batch_indexes)

        # print(images.shape, masks.shape, depths.shape)
        video_name = f"video_{self.record}_{start_idx}_{start_idx + self.num_frames}"




        # print("prompt", prompt, type(prompt)) # depth = torch.tensor(depth)
        return dict(image=torch.tensor(images), mask=torch.tensor(masks), depth = torch.tensor(depths), video_name =video_name)





class DatasetRetinaVideo(Dataset):
    def __init__(self, image_size = 256, from_file = [], num_frames = 1):
        self.data = []
        # with open("/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/Cataract_1K_seg_full.json", 'rt') as f:
        #     for line in f:
        #         self.data.append(json.loads(line))



        # self.df = pd.read_csv("/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/GFG_test.csv")
        # print(self.df.columns)
        # print(self.df.head(3))

        self.image_shape = (image_size, image_size)

        if len(from_file) > 0:


            with open(from_file, "rb") as f:
                self.image_paths = pickle.load(f)


        print(f"Loaded {len(self.image_paths)} samples")


    def __len__(self):
        return len(self.image_paths) 


    def load_image(self, path, img_type):

        if img_type == "image":

            image = Image.open(path)

            image = np.asarray(image)
            image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)

            image = image.astype(np.float16).transpose(2,0,1)
            # image = image.astype(np.float16)
            # image = image.transpose(2,0,1)
            # image = np.moveaxis(image, -1, 0)
            image = image / 255.


        elif img_type == "mask":


            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale


            mask = np.asarray(mask)
            mask = cv2.resize(mask, self.image_shape, interpolation=cv2.INTER_NEAREST)


            # segmentation = np.eye(14)[mask].astype(np.float32) #.transpose(2,0,1)

            # segmentation = segmentation.astype(np.float16)


            # # print(image.shape, segmentation.shape)
            # segmentation = segmentation.transpose(2,0,1)

            image = mask

        return image


    def get_batch(self, paths):

        # base_idx = idx % len(self.df)
        image_list = []
        mask_list = []

        images_paths = paths[:len(paths)//2]
        masks_paths = paths[len(paths)//2:]

        # print(len(images_paths), len(masks_paths))
        # print(images_paths[0], masks_paths[0])

        for path in images_paths:


            image = self.load_image(path, img_type ="image")

            image_list.append(image)


        for path in masks_paths:


            mask = self.load_image(path, img_type = "mask")

            mask_list.append(mask)

        images = np.stack(image_list)
        masks = np.stack(mask_list)

        return images, masks



    def __getitem__(self, idx):
        idxs = self.image_paths[idx]
        # print("idxs ", len(idxs))

        images, masks = self.get_batch(idxs)


        return dict(image=torch.tensor(images), mask=torch.tensor(masks), depth = "", video_name="")











def preprocess_size(image1, image2, padding_factor=32):
    '''
        img: [b, c, h, w]
    '''
    transpose_img = False
    # the model is trained with size: width > height
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    # inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
    #                 int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        
    inference_size = [256,256]

    assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)
    
    return image1, image2, inference_size, ori_size, transpose_img


def postprocess_size(flow_pr, inference_size, ori_size, transpose_img):

    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                align_corners=True)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)
    
    return flow_pr


# @torch.no_grad()
def get_optical_flows(unimatch, video_frame):
    '''
        video_frame: [b, t, c, w, h]
    '''

    video_frame = video_frame * 255

    # print(video_frame.dtype)

    flows = []
    for i in range(video_frame.shape[1] - 1):
        image1, image2 = video_frame[:, 0], video_frame[:, i + 1]
        # print(image1.dtype)
        image1_r, image2_r, inference_size, ori_size, transpose_img = preprocess_size(image1, image2)
        # print(image1_r.dtype)
        results_dict_r = unimatch(image1_r, image2_r,
            attn_type='swin',
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1],
            num_reg_refine=6,
            task='flow',
            pred_bidir_flow=False,
            )
        flow_r = results_dict_r['flow_preds'][-1]  # [b, 2, H, W]
        # print(flow_r.shape)
        flow = postprocess_size(flow_r, inference_size, ori_size, transpose_img)
        flows.append(flow.unsqueeze(1))  # [b, 1, 2, w, h]
    
    flows = torch.cat(flows, dim=1).to(torch.float16)  # [b, t, 2, w, h]
    return flows


def get_dav2_flows(depth_anything, pixel_values, write=False):

    depth_frames = []
    depth_imgs = []
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # print(f'pixel_values::: {pixel_values.shape}')
    # pdb.set_trace()
    for idx in range(pixel_values.shape[0]):
        image = pixel_values[idx,0]
        image = image.clone().detach().cpu().numpy() 
        image = image.transpose(1, 2, 0)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)  # Normalize if necessary

        # raw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        depth = np.expand_dims(depth_anything.infer_image(image, 256), axis=(0))
        
        depth_img = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_img = depth_img.astype(np.uint8)
        depth_img = np.repeat(depth_img[..., np.newaxis], 3, axis=-1)
        depth_frames.append(depth)
        depth_imgs.append(depth_img)
        if write:
            pass
    depth_imgs = np.array(depth_imgs)
    depth_frames = torch.tensor(np.array(depth_frames))
    # print(f'depth_frames shape : {depth_frames.shape}')
    # print(f'depth_frame shape : {depth.shape}')
    return depth_frames.cuda(), depth_imgs # [b, f-1, 3, h, w]

def get_sam_flows(predictor, pixel_values):
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    # print(f'pixel_values::: {pixel_values.shape}')
    # pdb.set_trace()
    for idx in range(pixel_values.shape[0]):
        image = pixel_values[idx,0]
        image = image.clone().detach().cpu().numpy() 
        image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)  # Normalize if necessary
        predictor.set_image(image)
        seg_feature = predictor.features # tensor [1,256,64,64]
        if idx ==0:
            seg_features = seg_feature  
        else:
            seg_features = torch.cat((seg_features, seg_feature), dim=0) 
    

    return seg_features

def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = torch.utils.data.DataLoader(
            dataset= sample_dataset,
            batch_size=sample_size,
            drop_last=True
        )

        for item in sample_loader:
            yield item


def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data


min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5


def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )


    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""

    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")


    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)

"""Use gaussian bluring"""
def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out

"""Use VAE to encode tensor to latent space"""
def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=21,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )


    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        default=False,
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    
    parser.add_argument(
        "--gpu_number",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
    )


    parser.add_argument(
        "--tool_loss_weight",
        type=float,
        default=0,
    )




    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():

    setup_cuda(use_memory_fraction=0.8, num_threads=16, visible_devices="0", multiGPU=False)

    args = parse_args()

    # setup_cuda(use_memory_fraction=0.99, num_threads=16, visible_devices=str(args.gpu_number), multiGPU=False)



    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    print(f"Accelerator using {accelerator.num_processes} processes, device: {accelerator.device}")


    generator = torch.Generator(
        device=accelerator.device).manual_seed(23123134)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )




    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = DualFlowControlNet_traj.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = DualFlowControlNet_traj.from_unet(unet)

    print("2 ====")
        
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)





    # Define Unimatch for optical flow prediction
    unimatch = UniMatch(feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task='flow').to('cuda')

    # checkpoint = torch.load('./Training/train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
    checkpoint = torch.load('train_utils/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')

    unimatch.load_state_dict(checkpoint['model'])

    unimatch = unimatch.to(accelerator.device).eval()
    unimatch.requires_grad_(False)

    # Depth Anything Model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vitb'])
    depth_anything.load_state_dict(torch.load(f'train_utils/dav2/ckpts/depth_anything_v2_vitb.pth', map_location='cpu'))
    depth_anything = depth_anything.to(accelerator.device).eval()
    depth_anything.requires_grad_(False)
    # ------Segment Mask Model-----
    # sam = sam_model_registry["vit_h"](checkpoint="sam2/checkpoints/sam_vit_h_4b8939.pth")
    # sam.to(accelerator.device).eval()
    # sam.requires_grad_(False)

    # predictor = SamPredictor(sam)



    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    #controlnet.to(accelerator.device, dtype=weight_dtype)
    # Create EMA for the unet.
    if args.use_ema:
        ema_controlnet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "controlnet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()

                if model.__class__.__name__ == "DualFlowControlNet_traj":
                    load_model = DualFlowControlNet_traj.from_pretrained(
                        input_dir, subfolder="controlnet"
                    )

                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())

            del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Optimize controlnet
    controlnet.requires_grad_(True)
    parameters_list = []

    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check para
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in controlnet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()
    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    # train_dataset = WebVid10M(
    #     sample_stride=args.sample_stride,
    #     sample_n_frames=args.num_frames, 
    #     sample_size=[args.height, args.width]
    #     )
    
    # TODO: change to our dataset
    # train_dataset = ESDDataset(
    #     meta_path='./dataset/condition_train_dual.csv',
    #     data_dir='./dataset/result_frame',
    #     sample_stride=args.sample_stride,
    #     sample_n_frames=args.num_frames, 
    #     sample_size=[args.height, args.width]
    #     )

    train_dataset = DatasetMultimodalVideoSimToReal(image_size=256,
                                    main_path=f'/home/MichalMo/projects/ControlNet-diffusers/records_cadis_1/records_single_files',
                                    record = "r2", num_frames= args.num_frames, limit_frames = None)

    # train_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data.pkl')
    # train_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data_train_789_30percent.pkl')

    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # test_dataset = ESDDataset(
    #     meta_path='./dataset/condition_test_dual.csv',
    #     data_dir='./dataset/result_frame',
    #     sample_size=[args.height, args.width],
    #     sample_n_frames=args.num_frames, 
    #     sample_stride=args.sample_stride
    #     )

    test_dataset = DatasetMultimodalVideoSimToReal(image_size=256,
                                main_path=f'/home/MichalMo/projects/ControlNet-diffusers/records_cadis_1/records_single_files',
                                record = "r2", num_frames= args.num_frames, limit_frames = None)
    
    # test_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data_test.pkl')
    # test_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data_test_789_30percent.pkl')

    # TODO: change to our dataset

    test_loader = create_iterator(1, test_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )


    unet, optimizer, lr_scheduler, train_dataloader, controlnet = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader, controlnet
    )


    if args.use_ema:
        ema_controlnet.to(accelerator.device)


    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))


    checkpointing_steps = len(train_dataloader)
    validation_steps = len(train_dataloader)


    print("3 ====")


    # if args.tool_loss_weight > 0:
    #     tool_loss_weight = args.tool_loss_weight


    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    def get_x0_from_noise_edm(noisy_latents, model_pred, sigmas):
        """
        Matches your specific EDM/Karras implementation.
        noisy_latents: [B, T, C, H, W] - The unnormalized noisy latents
        model_pred: [B, T, C, H, W] - The output from your UNet
        sigmas: [B, T, 1, 1, 1] - The reshaped sigma values
        """
        # print("get_x0_from_noise_edm")
        actual_noisy_part = noisy_latents[:, :, :model_pred.shape[2], :, :]
        # print(noisy_latents.shape, model_pred.shape, sigmas.shape, actual_noisy_part)
        # These constants must match your training loop exactly
        c_out = -sigmas / ((sigmas**2 + 1)**0.5)
        c_skip = 1 / (sigmas**2 + 1)
        # print(c_out.shape, c_skip.shape)
        # This is your 'denoised_latents' (the x0 estimate)
        denoised_latents = model_pred * c_out + c_skip * actual_noisy_part
        return denoised_latents

    """normalize image using CLIPImageProcessor 
    and get image embeddings using CLIPVisionModelWithProjection"""
    def encode_image(pixel_values):
        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings= image_embeddings.unsqueeze(1)
        return image_embeddings

    """motion_bucket_ids? adding time embeddings? unet.add_embedding.linear_1.in_features?"""
    def _get_add_time_ids(
        fps, # frames per second
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength, # noise strength
        dtype,
        batch_size,
        unet=None,
    ):

        motion_bucket_ids = torch.tensor([motion_bucket_ids], dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    
        # Check for batch size consistency
        if motion_bucket_ids.size(0) != batch_size:
            raise ValueError("The length of motion_bucket_ids must match the batch_size.")
    
        add_time_ids = [fps, noise_aug_strength]
    
        # Concatenate fps and noise_aug_strength with motion_bucket_ids along the second dimension
        add_time_ids = torch.tensor(add_time_ids, dtype=dtype).repeat(batch_size, 1)
        add_time_ids = torch.cat([add_time_ids, motion_bucket_ids.to(add_time_ids)], dim=1)
    
        # Checking the dimensions of the added time embedding
        passed_add_embed_dim = unet.config.addition_time_embed_dim * add_time_ids.size(1)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. "
                "Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )
    
        return add_time_ids

    global_min_loss = 100000


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.base_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.base_dir, path), strict=False)
            print("Accelerator loaded")
            # assert False/
            # global_step = int(path.split("-")[1])

            # resume_global_step = global_step * args.gradient_accumulation_steps
            # first_epoch = global_step // num_update_steps_per_epoch

            # # resume_step = resume_global_step % (
            # #     num_update_steps_per_epoch * args.gradient_accumulation_steps)
            # print("first_epoch ", first_epoch, global_step)
            # assert False

            first_epoch = int(path.split("-")[1])

            gs = path.split("-")[-1]
            df1 = pd.read_csv(f"{args.base_dir}training_log.csv", index_col=0)
            global_min_loss = df1[df1["global_step"]== int(gs)]["mean_epoch_loss"].values[0]
            print(f"Loaded Mean global LOSS: {global_min_loss} from checkpoint: {gs}")




    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    global_start_time = time.time()
    cumulative_time = 0.0

    loss_fn_lpips = lpips.LPIPS(net='vgg').to("cuda")

    loss_fn_lpips.eval()


    csv_log_path = os.path.join(args.output_dir, "training_log.csv")
    if accelerator.is_main_process and not os.path.exists(csv_log_path):
        with open(csv_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "global_step", "mean_epoch_loss", "mean_epoch_loss_diff", "mean_epoch_loss_warp", "mean_diff_lpips", "lr", "epoch_duration", "cumulative_time"])


    args_save_path = os.path.join(args.output_dir, "parameters.txt")
    with open(args_save_path, 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    controlnet.mask_weight = 5.0
    print("controlnet.mask_weight ", controlnet.mask_weight)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    lora_path = "/home/MichalMo/projects/SurgSora/kohya_ss/outputs/cataract2-000026.safetensors"
    lora_sd = load_file(lora_path,  device="cuda")

    lora = LoRANetwork(
        text_encoder=None,
        unet=unet
    ).to(accelerator.device)

    lora.load_state_dict(lora_sd, strict=False)

    lora.apply_to(None, unet)

    unet.to(accelerator.device)
    controlnet.unet = unet
    controlnet.to(accelerator.device)

    for module in lora.modules():
        module.to(accelerator.device)

    for name, param in controlnet.unet.named_parameters():
        if param.device.type != "cuda":
            print("CPU param:", name)
            break




    final_epoch = first_epoch+args.num_train_epochs

    for epoch in range(first_epoch, final_epoch):


        epoch_start_time = time.time()
        epoch_losses = []
        epoch_losses_diff = []
        epoch_losses_warp = []
        epoch_losses_lpips = []


        controlnet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            # print(step)
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(controlnet):

                pixel_values = batch["image"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                masks = batch["mask"].float()

                # bbox = batch["bbox"].to(weight_dtype).to(
                #     accelerator.device, non_blocking=True
                # )

                # convert to latent representation
                # print(pixel_values.shape)
                # pixel_values = pixel_values[0]
                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # Getting SNR for each timestep
                sigmas = rand_cosine_interpolated(shape=[bsz,], 
                                                  image_d=image_d, 
                                                  noise_d_low=noise_d_low, 
                                                  noise_d_high=noise_d_high,
                                                  sigma_data=sigma_data, 
                                                  min_value=min_value, 
                                                  max_value=max_value).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                    
                train_noise_aug = 0.01
                # also getting the first frame as conditional latents
                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = small_noise_latents[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                noisy_latents  = latents + noise * sigmas_reshaped

                # get timesteps using sigma (SNR)
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(latents.device)
                
                # normalize the noisy latents
                inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                

                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :].float())

                added_time_ids = _get_add_time_ids(
                    7, # frame per second
                    70, # motion buket ids: the motion bucket id to use for the generated video. This can be used to control the motion of the generated video. Increasing the motion bucket id increases the motion of the generated video.
                    train_noise_aug, # noise_aug_strength == 0.0
                    encoder_hidden_states.dtype, # encoded image using CLIP
                    bsz, # batch size
                    unet
                )
                added_time_ids = added_time_ids.to(latents.device)

                if args.conditioning_dropout_prob is not None: # default 0.1
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)

                # `inp_noisy_latents` contains 2 types of latents
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2).to(torch.float16)
                
                # get optical flows via unimatch
                # TODO: change the optical flow to depth information
                depths,_ = get_dav2_flows(depth_anything, pixel_values) 
                depths = depths.to(torch.float16).to(accelerator.device)

                # Baseline
                # flows = get_optical_flows(unimatch, pixel_values) 

                # Consecutive masked blur flows
                # flows = get_consecutive_optical_flows(unimatch.cuda(), pixel_values)
                # flows = apply_masked_blur_to_flow(flows.to(torch.float32), masks.unsqueeze(2)[:,1:,...] )

                # Finetune warp_with_mask loss
                flows = get_consecutive_optical_flows(unimatch.cuda(), pixel_values)


                ###############################################################
                # depths = batch["depth"].float()
                # depths = depths[:,0,...] # get first depth frame


                # Inside your training loop
                my_gt_mask = batch['mask']  # Shape: [B, 14, 256, 256]
                # 1. Resize to match DSI spatial resolution (e.g., 64x64)
                dsi_mask_spatial = F.interpolate(my_gt_mask, size=(64, 64), mode='nearest')
                # print("dsi_mask_spatial shape ", dsi_mask_spatial.shape, dsi_mask_spatial.dtype, dsi_mask_spatial.device)

                # 2. If the model expects 256 channels (SAM feature size), 
                # you can pad or project. A common trick is to repeat classes:
                masks = F.pad(dsi_mask_spatial, (0, 0, 0, 0, 0, 256 - args.num_frames)).to(torch.float16).to(accelerator.device) # Pad 14 to 256
                ###############################################################


                # masks = torch.ones([2,1,64,64]).to(torch.float16).to(accelerator.device)
                # depths = torch.ones([2,1,256,256]).to(torch.float16).to(accelerator.device)

                # print("images shape ", pixel_values.shape, pixel_values.dtype, pixel_values.device)
                # print("masks shape ", masks.shape, masks.dtype, masks.device)

                # print("flows shape ", flows.shape, flows.dtype, flows.device)
                # print("depths shape ", depths.shape, depths.dtype, depths.device)

                # print("inp_noisy_latents shape ", inp_noisy_latents.shape, inp_noisy_latents.dtype, inp_noisy_latents.device)




                # print("encoder_hidden_states dtype ", encoder_hidden_states.dtype, encoder_hidden_states.device)
                # print("added_time_ids dtype ", added_time_ids.dtype, added_time_ids.device)


                # the first frame of the video
                controlnet_image = pixel_values[:, 0, :, :, :]
                # print(f'controlnet_image shape is {controlnet_image.shape}')
                # print(torch.min(controlnet_image), torch.max(controlnet_image))
                # print("controlnet_image ", controlnet_image.shape)

                target = latents

                down_block_res_samples, mid_block_res_sample, _, _ = controlnet(
                    inp_noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controlnet_image,  # [b, c, W, H] first frame of the video
                    controlnet_flow=flows,          # [b,f-1,c-1, W, H]
                    controlnet_mask=masks,           # [b, 256, 64, 64]
                    controlnet_depth=depths,        # [b,1,256,256]
                    return_dict=False,
                    conditioning_scale = 0.85
                )
            
                # Predict the noise residual
                model_pred = unet(
                    inp_noisy_latents, 
                    timesteps, 
                    encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                sigmas = sigmas_reshaped
                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)


                # # MSE loss
                # loss = torch.mean(
                #     (weighing.float() * (denoised_latents.float() -
                #      target.float()) ** 2).reshape(target.shape[0], -1),
                #     dim=1,
                # )
                # print("loss ", loss.shape)
                # loss = loss.mean()

                # epoch_losses.append(loss.detach().item())




                # 1. Setup your shapes
                # masks: [1, 21, 256, 256]
                # model_pred: [1, 21, 4, 32, 32]
                B, F_frames, H_pixel, W_pixel = my_gt_mask.shape
                _, _, C_latent, H_latent, W_latent = model_pred.shape

                # 2. Add a channel dimension and flatten Batch + Frames
                # Interpolate expects 4D: [Batch, Channels, H, W]
                # We treat (B * F) as the new batch size
                masks_4d = my_gt_mask.view(B * F_frames, 1, H_pixel, W_pixel).float()

                dilated_masks = F.max_pool2d(masks_4d, kernel_size=2, stride=1, padding=1)
                dilated_latent_masks = F.interpolate(
                    dilated_masks, 
                    size=(H_latent, W_latent), 
                    mode='nearest'
                )
                dilated_latent_masks = dilated_latent_masks.view(B, F_frames, 1, H_latent, W_latent)
                # dilated_weight_map = 1.0 + (dilated_latent_masks > 4).float() * args.tool_loss_weight  # Tool = 10x weight


                # MSE loss Baseline
                # loss = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)

                # loss_w = dilated_weight_map * loss

                # loss_w = torch.mean(loss_w.reshape(target.shape[0], -1), dim = 1)
                # loss_w = loss_w.mean()

                # epoch_losses.append(loss_w.detach().item())
                ####################################

                # LOSS WITH WRAP

                # loss = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
                # loss_w = dilated_weight_map * loss
                # loss_w = torch.mean(loss_w.reshape(target.shape[0], -1), dim = 1)
                # loss_w = loss_w.mean()

                pred_x0 = get_x0_from_noise_edm(inp_noisy_latents, model_pred, sigmas_reshaped)

                ratio = 8.0 
                latent_flow = F.interpolate(
                    flows.flatten(0, 1), 
                    size=(32, 32), 
                    mode='bilinear'
                ).view(B, F_frames-1, 2, 32, 32) / ratio


                # with torch.no_grad():
                #     print("--- SCALE DIAGNOSTICS ---")
                #     print(f"Pixels: Range [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")
                #     print(f"Latents (pred_x0): Std {pred_x0.std():.2f}, Mean {pred_x0.mean():.2f}")
                    
                #     # Calculate flow magnitude (hypotenuse of U and V)
                #     flow_mag = torch.norm(latent_flow, dim=2) 
                #     print(f"Flow Magnitude (Pixel Space): Max {flow_mag.max():.2f}, Mean {flow_mag.mean():.2f}")
                    
                #     # Check the resolution ratio
                #     ratio = pixel_values.shape[-1] / pred_x0.shape[-1]
                #     print(f"Downsampling Ratio: {ratio}x")
                    
                #     if flow_mag.max() > pred_x0.shape[-1] and ratio > 1:
                #         print("⚠️ WARNING: Flow magnitude is larger than latent resolution. Ensure you divide flow by ratio!")







                                                # loss_warp = 0
                                                # for t in range(pred_x0.shape[1] - 1):
                                                #     # Warp latents using the rescaled latent_flow
                                                #     # We use pred_x0 which is your denoised latent estimate
                                                #     warped_t_plus_1, valid_mask = warp_with_mask(pred_x0[:, t+1], latent_flow[:, t])
                                                    
                                                #     # Calculate L1 loss only on valid pixels (those not warped from outside)
                                                #     diff = torch.abs(warped_t_plus_1 - pred_x0[:, t])
                                                #     loss_warp += (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                                                # lambda_warp = 0.01
                                                # # total_loss = loss_w + (lambda_warp * loss_warp)


                                                # # LPIPS
                                                # # 1. Prepare prediction for decoding
                                                # # We only take a subset of frames (e.g., every 4th frame) to save 80% of VAE compute
                                                # # or decode all if your hardware allows.
                                                # B, F2, C, H, W = pred_x0.shape
                                                # pred_x0_for_lpips = pred_x0.view(-1, C, H, W).to(vae.dtype)

                                                # # 2. Decode to Pixel Space
                                                # # Standard VAEs output pixels in range [-1, 1]
                                                # # We use the scaling factor to bring it back to the range the VAE expects
                                                # # decoded_pixels = vae.decode(pred_x0_for_lpips / vae.config.scaling_factor, num_frames=F2).sample

                                                # decoded_step = 4
                                                # decoded_pixels = vae.decode(
                                                #     pred_x0_for_lpips[:, ::decoded_step] / vae.config.scaling_factor, 
                                                #     num_frames=pred_x0_for_lpips[:, ::step].shape[1]
                                                # ).sample
                                                # decoded_pixels = decoded_pixels.to(torch.float32)


                                                # # 3. Prepare the Ground Truth First Frame (Reference)
                                                # # pixel_values[:, 0] is [B, 3, 256, 256]. We repeat it for all F frames.
                                                # ref_frame_real = pixel_values[:, 0] # This is your realistic anchor
                                                # ref_expanded = ref_frame_real.unsqueeze(1).repeat(1, F2, 1, 1, 1).view(-1, 3, 256, 256)

                                                # if len(decoded_pixels.shape) == 5:
                                                #     decoded_pixels = decoded_pixels.view(-1, 3, 256, 256)

                                                # # 4. Calculate LPIPS
                                                # # This compares the 'look' of the prediction to the 'look' of frame 0
                                                # loss_lpips = loss_fn_lpips(decoded_pixels, ref_expanded).mean()



                                                # target_frame_0 = latents[:, 0].unsqueeze(1).repeat(1, F2, 1, 1, 1)

                                                # # Now your MSE (loss_w) will try to keep the textures looking like Frame 0
                                                # loss_w = (weighing.float() * (denoised_latents.float() - target_frame_0.float()) ** 2)

                                                # # 5. Combine with your existing losses
                                                # # Scale lambda_lpips so it's roughly 1/10th of your MSE loss initially
                                                # lambda_lpips = 0.05 
                                                # total_loss = loss_w + (lambda_warp * loss_warp) + (lambda_lpips * loss_lpips)



                loss_warp = 0
                for t in range(pred_x0.shape[1] - 1):
                    warped_t_plus_1, valid_mask = warp_with_mask(pred_x0[:, t+1], latent_flow[:, t])
                    diff = torch.abs(warped_t_plus_1 - pred_x0[:, t])
                    loss_warp += (diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

                B, F2, C, H, W = pred_x0.shape
                decoded_step = 4 

                # 1. Slice in 5D: [B, F, C, H, W] -> [B, F_sliced, C, H, W]
                pred_x0_sliced = pred_x0[:, ::decoded_step] 
                num_sliced_frames = pred_x0_sliced.shape[1]

                # 2. Flatten for the VAE's conv layers: [B * F_sliced, C, H, W]
                # This satisfies the "Expected 4D input" requirement
                pred_x0_flattened = pred_x0_sliced.reshape(-1, C, H, W)

                # 3. Decode
                # We pass the flattened tensor, but we still tell the VAE the num_frames
                # so it can internaly unflatten for temporal layers when needed
                decoded_output = vae.decode(
                    pred_x0_flattened.to(vae.dtype) / vae.config.scaling_factor, 
                    num_frames=num_sliced_frames
                ).sample.to(torch.float32)

                # 4. Final check: ensure it's 4D for LPIPS [B*F_sliced, 3, 256, 256]
                if len(decoded_output.shape) == 5:
                    decoded_pixels = decoded_output.view(-1, 3, 256, 256)
                else:
                    decoded_pixels = decoded_output

                # --- 5. Match Reference ---
                ref_frame_real = pixel_values[:, 0].to(torch.float32)
                ref_expanded = ref_frame_real.unsqueeze(1).repeat(1, num_sliced_frames, 1, 1, 1)
                ref_expanded = ref_expanded.view(-1, 3, 256, 256)
                # --- 4. Calculate LPIPS ---
                loss_lpips = loss_fn_lpips(decoded_pixels, ref_expanded).mean()

                # target_frame_0 = latents[:, 0].unsqueeze(1).repeat(1, F2, 1, 1, 1)
        
                # # print("loss_w ", loss_w.shape, loss_w) # orch.Size([2, 21, 4, 32, 32])
                # # print("loss_lpips ", loss_lpips.shape, loss_lpips) # [2, 21, 4, 32, 32]
                # # print("loss_warp ", loss_warp.shape, loss_warp) 
                # # print("dilated_weight_map ", dilated_weight_map.shape)
                # dilated_weight_map = 1.0 - (dilated_latent_masks > 4).float()

                # loss_w = dilated_weight_map * loss_w
                # loss_w = torch.mean(loss_w.reshape(target.shape[0], -1), dim = 1).mean()

                dilated_weight_map_binary = (dilated_latent_masks> 0.5).float()

                target_frame_0 = latents[:, 0].unsqueeze(1).repeat(1, F2, 1, 1, 1)
                composite_target = (latents * dilated_weight_map_binary) + (target_frame_0 * (1 - dilated_weight_map_binary))
                loss_w = (weighing.float() * (denoised_latents.float() - composite_target.float()) ** 2)
                
                background_weight = 1.0
                tool_weight = 0.5 # Lower so it doesn't copy the "sim" texture too hard
                weight_map = (1 - dilated_weight_map_binary) * background_weight + (dilated_weight_map_binary * tool_weight)

                loss_w = (loss_w * weight_map).mean()





                # Updated Lambdas for stability
                lambda_mse = 0.001 # To counter the ~1000 range
                lambda_warp = 0.1  # Increased from 0.01
                lambda_lpips = 1.0 # Increased from 0.05

                total_loss = (lambda_mse * loss_w.mean()) + (lambda_warp * loss_warp) + (lambda_lpips * loss_lpips)

                # print(total_loss.shape, total_loss)


                # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = accelerator.gather(
                #     total_loss.repeat(args.per_gpu_batch_size)).mean()
                avg_loss = accelerator.gather( total_loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "diff_loss": loss_w, "warp_loss": loss_warp, "loss_lpips": loss_lpips}, step=global_step)

   

                train_loss = 0.0

            logs = {"step_loss": total_loss.detach().item(), 
                    "diff_loss": loss_w.detach().item(), 
                    "warp_loss": loss_warp.detach().item(), 
                    "loss_lpips": loss_lpips.detach().item(), 

                    "lr": lr_scheduler.get_last_lr()[0]}
            
            progress_bar.set_postfix(**logs)


            epoch_losses.append(total_loss.detach().item())
            epoch_losses_diff.append(loss_w.detach().item())
            epoch_losses_warp.append(loss_warp.detach().item())
            epoch_losses_lpips.append(loss_lpips.detach().item())


            if global_step >= args.max_train_steps:
                break

        # End of step 

        # End of epoch
        if accelerator.is_main_process:
            with open(csv_log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                current_lr = lr_scheduler.get_last_lr()[0]
                epoch_duration = time.time() - epoch_start_time
                mean_epoch_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else float("nan")

                mean_epoch_loss_diff = np.mean(epoch_losses_diff) if len(epoch_losses) > 0 else float("nan")
                mean_epoch_loss_warp = np.mean(epoch_losses_warp) if len(epoch_losses) > 0 else float("nan")
                mean_epoch_loss_lpips = np.mean(epoch_losses_lpips) if len(epoch_losses) > 0 else float("nan")

                cumulative_time = time.time() - global_start_time

                writer.writerow([epoch, global_step, mean_epoch_loss, mean_epoch_loss_diff, mean_epoch_loss_warp, mean_epoch_loss_lpips, current_lr, epoch_duration, cumulative_time])

            logger.info(f"[LOG] Epoch {epoch}, Step {global_step}: "
                    f"mean_loss={mean_epoch_loss:.6f}, global_mean_loss={global_min_loss:.6f}, "
                      f"mean_diff_loss={mean_epoch_loss_diff:.6f}, mean_warp_loss={mean_epoch_loss_warp:.6f}, mean_diff_lpips={mean_epoch_loss_lpips:.6f}, lr={current_lr:.2e}, "
                    f"epoch_time={epoch_duration:.1f}s, total_time={cumulative_time/60:.1f}min")



            if mean_epoch_loss < global_min_loss:
                global_min_loss = mean_epoch_loss

                # save checkpoints!
                if global_step % (checkpointing_steps) == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [
                            d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(
                            checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(
                                checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(
                                f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(
                                    args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    ##############################################

                # --- SAVE PIPELINE ---
                if accelerator.is_main_process:
                    controlnet_unwrapped = accelerator.unwrap_model(controlnet)
                    unet_unwrapped = accelerator.unwrap_model(unet)
                    image_encoder_unwrapped = accelerator.unwrap_model(image_encoder)
                    vae_unwrapped = accelerator.unwrap_model(vae)

                    pipeline = DualFlowControlNetPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unet_unwrapped,
                        controlnet=controlnet_unwrapped,
                        image_encoder=image_encoder_unwrapped,
                        vae=vae_unwrapped,
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )

                    # pipeline_save_path = os.path.join(args.output_dir, f"pipeline-{global_step}")
                    pipeline.save_pretrained(args.output_dir)

                    logger.info(f"Pipeline saved to {args.output_dir}")


                # sample images!
                if (
                    (global_step % (validation_steps) == 0)
                    or (global_step == 1)
                ):
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} videos."
                    )
                    # create pipeline
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_controlnet.store(controlnet.parameters())
                        ema_controlnet.copy_to(controlnet.parameters())
                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = DualFlowControlNetPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        controlnet=accelerator.unwrap_model(
                            controlnet),
                        image_encoder=accelerator.unwrap_model(
                            image_encoder),
                        vae=accelerator.unwrap_model(vae),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)



                    # run inference
                    val_save_dir = os.path.join(
                        args.output_dir, "validation_images")



                    if not os.path.exists(val_save_dir):
                        os.makedirs(val_save_dir)


                    with torch.autocast(
                        str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                    ):
                        for val_img_idx in range(args.num_validation_images):

                            val_batch = next(test_loader)

                            val_pixel_values = val_batch['image'].to(weight_dtype).to(
                                accelerator.device
                            )  # [b, t, c, W, H]

                            val_masks_org = val_batch['mask']  # Shape: [B, 14, 256, 256]

                            # print("val_pixel_values ", val_pixel_values.shape)
                            val_depths,depth_img = get_dav2_flows(depth_anything, val_pixel_values)
                            val_depths = val_depths.to(torch.float16).to(accelerator.device)

                            # val_flows = get_optical_flows(unimatch, val_pixel_values) 
                            # val_flows = val_flows.to(torch.float16).to(accelerator.device)

                            # val_masks = get_sam_flows(predictor, val_pixel_values)
                            # val_flows = torch.zeros([2,20,2,256,256]).to(torch.float16).to(accelerator.device)

                            # val_flows = get_consecutive_optical_flows(unimatch.to(accelerator.device), val_pixel_values).to(accelerator.device)
                            # val_flows = apply_masked_blur_to_flow(val_flows.to(torch.float32), val_masks_org.unsqueeze(2)[:,1:,...].to(accelerator.device) ).to(accelerator.device)

                            val_flows = get_consecutive_optical_flows(unimatch.cuda(), val_pixel_values)
                            val_flows = val_flows.to(torch.float16).to(accelerator.device)


                            # 1. Resize to match DSI spatial resolution (e.g., 64x64)
                            dsi_mask_spatial = F.interpolate(val_masks_org, size=(64, 64), mode='nearest')

                            # 2. If the model expects 256 channels (SAM feature size), 
                            # you can pad or project. A common trick is to repeat classes:
                            val_masks = F.pad(dsi_mask_spatial, (0, 0, 0, 0, 0, 256 - 21)).to(torch.float16).to(accelerator.device) # Pad 14 to 256
                            val_masks = val_masks[0,...].unsqueeze(0)
                            # print(f"depths.shape is {depths.device} ,flows.shape = {flows.device}")



                            val_controlnet_image = val_pixel_values[:, 0:1, :, :, :].repeat(1, val_pixel_values.shape[1], 1, 1, 1)  # [1,21,3,384,384]

                            val_controlnet_depth = np.repeat(depth_img, val_pixel_values.shape[1], axis=1)

                            pil_val_pixel_values = [Image.fromarray((val_pixel_values[0][i].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) for i in range(val_pixel_values.shape[1])]
                            



                            # For testing

                            # val_flows = torch.ones([1,20,2,256,256]).to(torch.float16).to(accelerator.device)
                            # val_depths = torch.ones([1,1,256,256]).to(torch.float16).to(accelerator.device)
                            # val_masks = torch.ones([1,256,64,64]).to(torch.float16).to(accelerator.device)





                            pil_val_pixel_values_numpy = np.asarray(pil_val_pixel_values[0])
                            # print("pil_val_pixel_values_numpy ", len(pil_val_pixel_values), pil_val_pixel_values_numpy.shape, pil_val_pixel_values_numpy.dtype)

                            # print("val_masks shape ", val_masks.shape, val_masks.dtype, val_masks.device)

                            # print("val_flows shape ", val_flows.shape, val_flows.dtype, val_flows.device)
                            # print("val_depths shape ", val_depths.shape, val_depths.dtype, val_depths.device)

                            # print("inp_noisy_latents shape ", inp_noisy_latents.shape, inp_noisy_latents.dtype, inp_noisy_latents.device)


                            # print("encoder_hidden_states dtype ", encoder_hidden_states.dtype, encoder_hidden_states.device)
                            # print("added_time_ids dtype ", added_time_ids.dtype, added_time_ids.device)

                            # val_masks = torch.cat([val_masks, val_masks], dim=0)
                            # val_depths = torch.cat([val_depths, val_depths], dim=0)
                            # val_flows = torch.cat([val_flows, val_flows], dim=0)


                            num_frames = args.num_frames
                            video_frames = pipeline(
                                pil_val_pixel_values[0], 
                                pil_val_pixel_values[0],
                                val_flows,              #  [b,f-1,c-1, W, H]
                                val_depths,             # [b,1,256,256]
                                val_masks,              # [b, 256, 64, 64]
                                height=args.height,
                                width=args.width,
                                num_frames=num_frames,
                                decode_chunk_size=8,
                                motion_bucket_id=70,
                                fps=7,
                                noise_aug_strength=0.01,
                                controlnet_cond_scale = 0.85

                                # generator=generator,
                            ).frames[0]

                            for i in range(num_frames):
                                img = video_frames[i]
                                video_frames[i] = np.array(img)
                            viz_flows = []
                            for i in range(val_flows.shape[1]):
                                temp_flow = val_flows[0][i].permute(1, 2, 0)
                                viz_flows.append(flow_to_image(temp_flow))
                            viz_flows = [np.uint8(np.ones_like(viz_flows[-1]) * 255)] + viz_flows
                            viz_flows = np.stack(viz_flows)  # [t-1, h, w, c]
                            flow_nps = viz_flows

                            masks_nps = grayscale_to_color_mask(val_masks_org[0].numpy(), "cataract")


                            out_nps = video_frames
                            gt_nps = (val_pixel_values[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
                            ctrl_nps = (val_controlnet_image[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
                            depth_nps =  (val_controlnet_depth[0]).astype(np.uint8)


                            out_nps_overlayed = overlay_segmentation(np.array(out_nps), masks_nps)
                            gt_nps_overlayed = overlay_segmentation(np.array(gt_nps), masks_nps)

                            # ctrl_nps - first frame for controlnet,
                            # flow_nps - viualization of flows
                            # depth_nps - depths
                            # out_nps - output video 
                            # gt_nps - ground truth video
                            # total_nps = np.concatenate([ctrl_nps, flow_nps, depth_nps, out_nps, gt_nps], axis=2)
                            total_nps = np.concatenate([ctrl_nps, flow_nps, masks_nps, depth_nps, out_nps, out_nps_overlayed, gt_nps, gt_nps_overlayed], axis=2)


                            # video_name = val_batch['video_name'][0].replace('/', '_').split('.')[0]
                            video_name = val_batch['video_name']
                            total_path = os.path.join(val_save_dir,
                                f"step_{global_step}_val_img/{str(val_img_idx).zfill(3)}-{video_name}.mp4",
                            )
                            os.makedirs(os.path.dirname(total_path), exist_ok=True)
                            # print(total_path, total_nps.shape, total_nps.dtype)
                            # torchvision.io.write_video(total_path, total_nps, fps=7, video_codec='h264', options={'crf': '10'})

                            imageio.mimsave(
                                total_path,
                                total_nps,
                                fps=7
                            )
                                                            
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_controlnet.restore(controlnet.parameters())

    


    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     controlnet = accelerator.unwrap_model(controlnet)
    #     if args.use_ema:
    #         ema_controlnet.copy_to(controlnet.parameters())

    #     pipeline = DualFlowControlNetPipeline.from_pretrained(
    #         args.pretrained_model_name_or_path,
    #         image_encoder=accelerator.unwrap_model(image_encoder),
    #         vae=accelerator.unwrap_model(vae),
    #         unet=unet,
    #         controlnet=controlnet,
    #         revision=args.revision,
    #     )
    #     pipeline.save_pretrained(args.output_dir)

        # if args.push_to_hub:
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )
    accelerator.end_training()


if __name__ == "__main__":
    main()
