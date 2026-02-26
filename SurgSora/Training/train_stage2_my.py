
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

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import time
import accelerate
import numpy as np
import imageio
import json
import PIL
import csv
import pickle
from PIL import Image, ImageDraw
import torch
import torchvision
import torch.nn.functional as F
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

import datetime
import diffusers

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

# from train_utils.dataset import WebVid10M, ESDDataset

from models.unet_spatio_temporal_condition_controlnet import UNetSpatioTemporalConditionControlNetModel
from pipeline.pipeline import DualFlowControlNetPipeline
from models.svdxt_featureflow_forward_controlnet_s2d_fixcmp_norefine import CMP_demo, DualFlowControlNet_traj

from train_utils.dav2.depth_anything_v2.dpt import DepthAnythingV2
# from segment_anything import sam_model_registry, SamPredictor
from train_utils.unimatch.unimatch.unimatch import UniMatch
from train_utils.unimatch.utils.flow_viz import flow_to_image
from train_utils.sample_flow_utils import flow_sampler
import matplotlib
import torch.nn as nn
from torch.utils.data import Dataset

from test_gpu import setup_cuda

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

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
























# Not in stage1
@torch.no_grad()
def get_cmp_flow(cmp, frames, sparse_optical_flow, mask):

    b, t, c, h, w = frames.shape
    assert h == 384 and w == 384
    frames = frames.flatten(0, 1)  # [b*13, 3, 256, 256]
    sparse_optical_flow = sparse_optical_flow.flatten(0, 1)  # [b*13, 2, 256, 256]
    mask = mask.flatten(0, 1)  # [b*13, 2, 256, 256]

    cmp_flow, _ = cmp_run(cmp, frames, sparse_optical_flow, mask)  # [b*13, 2, 256, 256]
    cmp_flow = cmp_flow.reshape(b, t, 2, h, w)

    return cmp_flow, _

@torch.no_grad()
def get_dav2_flows(depth_anything, pixel_values, write=False):

    depth_frames = []
    depth_imgs = []

    for idx in range(pixel_values.shape[0]):
        image = pixel_values[idx,0]
        image = image.clone().detach().cpu().numpy() 
        image = image.transpose(1, 2, 0)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)  # Normalize if necessary

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

    return depth_frames.cuda(), depth_imgs # [b, f-1, 3, h, w]

@torch.no_grad()
def get_sam_flows(predictor, pixel_values):

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


@torch.no_grad()
def cmp_run(cmp, image, sparse, mask):
    dtype = image.dtype
    image = image * 2 - 1
    cmp_output = cmp.model.model(image.float(), torch.cat([sparse, mask], dim=1).float())
    flow = cmp.fuser.convert_flow(cmp_output)
    if flow.shape[2] != image.shape[2]:
        flow = nn.functional.interpolate(
            flow, size=image.shape[2:4],
            mode="bilinear", align_corners=True)

    return flow.to(dtype), cmp_output.to(dtype)  # [b, 2, h, w]


def get_cmpsample_mask(flows):
    fb, fl, fc, fh, fw = flows.shape
    masks = []
    for i in range(fb): # batch size
        temp_flow = flows[i, -1].permute(1, 2, 0).cpu().numpy()  # [h, w, 2]
        _, temp_mask = flow_sampler(temp_flow, ['grid', 'watershed'])
        masks.append(temp_mask)
    masks = torch.from_numpy(np.stack(masks, axis=0)).to(flows.device, flows.dtype)  # [b, h, w, 2]
    masks = masks.unsqueeze(1).repeat(1, fl, 1, 1, 1).permute(0, 1, 4, 2, 3)  # [b, l, 2, h, w]

    return masks


def sample_inputs(unimatch, pixel_values):

    flows = get_optical_flows_1t(unimatch, pixel_values) 

    fb, fl, fc, fh, fw = flows.shape
    pb, pl, pc, ph, pw = pixel_values.shape


    mask = get_cmpsample_mask(flows)
    sparse_optical_flow = flows * mask
    

    if ph != 384 or pw != 384:
        flows_384 = F.interpolate(flows.flatten(0, 1), (384, 384)).reshape(fb, fl, 2, 384, 384) 
        flows_384[:, :, 0] *= 384 / pw
        flows_384[:, :, 1] *= 384 / ph

        pixel_values_384 = F.interpolate(pixel_values.flatten(0, 1), (384, 384)).reshape(pb, pl, 3, 384, 384)

        mask_384 = get_cmpsample_mask(flows_384)
        sparse_optical_flow_384 = flows_384 * mask_384

    else:
        flows_384, pixel_values_384 = flows, pixel_values
        sparse_optical_flow_384, mask_384 = sparse_optical_flow, mask
    

    controlnet_image = pixel_values[:, 0, :, :, :]

    return controlnet_image, sparse_optical_flow, mask, flows, pixel_values_384, sparse_optical_flow_384, mask_384


def preprocess_size(image1, image2, padding_factor=32):
    '''
        img: [b, c, h, w]
    '''
    transpose_img = False
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True


        
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


@torch.no_grad()
def get_optical_flows_1t(unimatch, video_frame):
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


def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = torch.utils.data.DataLoader(
            dataset= sample_dataset,
            batch_size=sample_size,
            drop_last=True
        )

        for item in sample_loader:
            yield item 


# copy from https://github.com/crowsonkb/k-diffusion.git
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

    # Make sure it is odd
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

    # for even kernels we need to do asymmetric padding :(
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

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
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
        default=25,
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
        "--train_width",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--train_height",
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():


    setup_cuda(use_memory_fraction=0.4, num_threads=16, visible_devices="0,1", multiGPU=True)


    args = parse_args()

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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
     #   log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

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
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
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
        controlnet = DualFlowControlNet_traj.from_pretrained(
            args.controlnet_model_name_or_path,
            low_cpu_mem_usage=False,
            device_map=None
        )
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = DualFlowControlNet_traj.from_unet(unet)
        
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # TODO: debug for the DAV2 in cmp
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**model_configs['vitb'])
    # depth_anything.load_state_dict(torch.load(f'./Training/ckpts/dav2/depth_anything_v2_vitb.pth', map_location='cpu'))
    depth_anything.load_state_dict(torch.load(f'train_utils/dav2/ckpts/depth_anything_v2_vitb.pth', map_location='cpu'))

    depth_anything = depth_anything.to(accelerator.device).eval()
    depth_anything.requires_grad_(False)
    # ------Segment Mask Model-----
    # sam = sam_model_registry["vit_h"](checkpoint="./Training/train_utils/sam/sam_vit_h_4b8939.pth")
    # sam.to(accelerator.device).eval()
    # sam.requires_grad_(False)

    # predictor = SamPredictor(sam)
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
    unimatch.eval()
    unimatch.requires_grad_(False)


    cmp = CMP_demo(
        'MOFA-Video-Traj/models/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/config.yaml',
        42000
    ).to(accelerator.device)
    cmp.requires_grad_(False)

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
    # flow_criterion.to(accelerator.device, dtype=weight_dtype)
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
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = DualFlowControlNet_traj.from_pretrained(
                    input_dir, subfolder="controlnet")
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

    parameters_list = []

    # optimize the parameters unrelated to flow_encode and controlnet_cond_embedding
    for name, para in controlnet.named_parameters():
        if ('flow_encoder' in name or 'controlnet_cond_embedding' in name):
            para.requires_grad = False
        else:
            parameters_list.append(para)
            para.requires_grad = True
    
    optimizer = optimizer_cls(
        parameters_list,
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

    # assert False

    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    # train_dataset = DatasetMultimodalVideo(image_size=256,
    #                                 main_path=f'/home/MichalMo/projects/ControlNet-diffusers/records_cadis_1/records_single_files',
    #                                 record = "r2", num_frames= args.num_frames, limit_frames = None)
    train_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data.pkl')


    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )


    # test_dataset = DatasetMultimodalVideo(image_size=256,
    #                             main_path=f'/home/MichalMo/projects/ControlNet-diffusers/records_cadis_1/records_single_files',
    #                             record = "r2", num_frames= args.num_frames, limit_frames = None)
    

    test_dataset = DatasetRetinaVideo(image_size=256, from_file='/home/MichalMo/projects/SurGrID/datasets/Cataract-1K/data_test.pkl')

    checkpointing_steps = len(train_dataloader)
    validation_steps = len(train_dataloader)

    # test_dataset = ESDDataset(
    #     meta_path='./dataset/condition_test_dual.csv',
    #     data_dir='./dataset/test_frame',
    #     sample_size=[args.train_height, args.train_width],
    #     sample_n_frames=args.num_frames, 
    #     sample_stride=args.sample_stride
    #     )
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

    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader, controlnet = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader, controlnet
    )

    if args.use_ema:
        ema_controlnet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDXtend", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps


    checkpointing_steps = len(train_dataloader)
    validation_steps = len(train_dataloader)




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


    def _get_add_time_ids(
        fps,
        motion_bucket_ids,  # Expecting a list of tensor floats
        noise_aug_strength,
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
    
    # accelerator.print(f"Resuming from checkpoint {args.ckpt}")
    # accelerator.load_state(args.ckpt)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
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
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    global_start_time = time.time()
    cumulative_time = 0.0


    global_min_loss = 100000

        # Create CSV log file
    csv_log_path = os.path.join(args.output_dir, "training_log.csv")
    if accelerator.is_main_process and not os.path.exists(csv_log_path):
        with open(csv_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "global_step", "mean_epoch_loss", "lr", "epoch_duration", "cumulative_time"])


    args_save_path = os.path.join(args.output_dir, "parameters.txt")
    with open(args_save_path, 'w+') as f:
        json.dump(args.__dict__, f, indent=2)




    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()



        epoch_start_time = time.time()
        epoch_losses = []


        controlnet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            # controlnet.module.cmp_model.eval()

            with accelerator.accumulate(controlnet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                pixel_values = batch["image"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                # bbox = batch["bbox"].to(weight_dtype).to(
                #     accelerator.device, non_blocking=True
                # )
                # print("pixel_values ", pixel_values.shape)

                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                sigmas = rand_cosine_interpolated(shape=[bsz,], image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high,
                                                  sigma_data=sigma_data, min_value=min_value, max_value=max_value).to(latents.device)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas_reshaped = sigmas.clone()
                while len(sigmas_reshaped.shape) < len(latents.shape):
                    sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                    
                train_noise_aug = 0.02
                small_noise_latents = latents + noise * train_noise_aug
                conditional_latents = small_noise_latents[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                noisy_latents  = latents + noise * sigmas_reshaped
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(latents.device)

                inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                
                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(
                    pixel_values[:, 0, :, :, :].float())

                added_time_ids = _get_add_time_ids(
                    6,
                    # batch["motion_values"],
                    127,
                    train_noise_aug, # noise_aug_strength == 0.0
                    encoder_hidden_states.dtype,
                    bsz,
                    unet
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
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

                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)
                
                # get optical flows via unimatch

                # controlnet_image, sparse_optical_flow, \
                #     mask, unimatch_flow, pixel_values_384, sparse_optical_flow_384, mask_384 = sample_inputs(unimatch, pixel_values)

                # get dev2 via unimatch
                controlnet_image, sparse_optical_flow, \
                    mask, unimatch_flow, pixel_values_384, sparse_optical_flow_384, mask_384 = sample_inputs(unimatch, pixel_values)
                
                fb, fl, fc, fh, fw = sparse_optical_flow.shape

                # print(controlnet_sparse_flow.shape)

                controlnet_flow, _ = get_cmp_flow(
                    cmp, 
                    pixel_values_384[:, 0:1, :, :, :].repeat(1, fl, 1, 1, 1), 
                    sparse_optical_flow_384, 
                    mask_384
                )
                controlnet_depth,_ = get_dav2_flows(depth_anything, pixel_values)
                # controlnet_masks = get_sam_flows(predictor, pixel_values)



                masks = batch["mask"].float()

                # Inside your training loop
                masks = batch['mask']  # Shape: [B, 14, 256, 256]

                # 1. Resize to match DSI spatial resolution (e.g., 64x64)
                dsi_mask_spatial = F.interpolate(masks, size=(64, 64), mode='nearest')

                # 2. If the model expects 256 channels (SAM feature size), 
                # you can pad or project. A common trick is to repeat classes:
                controlnet_masks = F.pad(dsi_mask_spatial, (0, 0, 0, 0, 0, 256 - args.num_frames)).to(torch.float16).to(accelerator.device) # Pad 14 to 256





                if fh != 384 or fw != 384:
                    scales = [fh / 384, fw / 384]
                    controlnet_flow = F.interpolate(controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
                    controlnet_flow[:, :, 0] *= scales[1]
                    controlnet_flow[:, :, 1] *= scales[0]


                target = latents

                # print("controlnet_image shape ", controlnet_image.shape, controlnet_image.dtype, controlnet_image.device)

                # print("images shape ", pixel_values.shape, pixel_values.dtype, pixel_values.device)
                # print("masks shape ", controlnet_masks.shape, controlnet_masks.dtype, controlnet_masks.device)

                # print("flows shape ", controlnet_flow.shape, controlnet_flow.dtype, controlnet_flow.device)
                # print("depths shape ", controlnet_depth.shape, controlnet_depth.dtype, controlnet_depth.device)

                # print("inp_noisy_latents shape ", inp_noisy_latents.shape, inp_noisy_latents.dtype, inp_noisy_latents.device)
                # print("encoder_hidden_states shape ", encoder_hidden_states.shape, encoder_hidden_states.dtype, encoder_hidden_states.device)



                down_block_res_samples, mid_block_res_sample, controlnet_flow, cmp_output = controlnet(
                    inp_noisy_latents, timesteps, encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    controlnet_cond=controlnet_image,  # [b, c, 384, 384]
                    controlnet_flow=controlnet_flow,  # [b, 13, 2, 384, 384]
                    controlnet_depth=controlnet_depth,
                    controlnet_mask=controlnet_masks,
                    return_dict=False,
                )


                # print(torch.cuda.memory_summary())



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

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()
                epoch_losses.append(loss.detach().item())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # assert False

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_controlnet.step(controlnet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0


            logs = {"step_loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0]}
            
            progress_bar.set_postfix(**logs)


            epoch_losses.append(loss.detach().item())


            if global_step >= args.max_train_steps:
                break



        if accelerator.is_main_process:
            with open(csv_log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                current_lr = lr_scheduler.get_last_lr()[0]
                epoch_duration = time.time() - epoch_start_time
                mean_epoch_loss = np.mean(epoch_losses) if len(epoch_losses) > 0 else float("nan")
                cumulative_time = time.time() - global_start_time

                writer.writerow([epoch, global_step, mean_epoch_loss, current_lr, epoch_duration, cumulative_time])

            logger.info(f"[LOG] Epoch {epoch}, Step {global_step}: "
                    f"mean_loss={mean_epoch_loss:.6f}, global_mean_loss={global_min_loss:.6f}, lr={current_lr:.2e}, "
                    f"epoch_time={epoch_duration:.1f}s, total_time={cumulative_time/60:.1f}min")


            if mean_epoch_loss < global_min_loss:
                global_min_loss = mean_epoch_loss
                # save checkpoints!
                if global_step % checkpointing_steps == 0:
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



                # sample images!
                if (
                    (global_step % validation_steps == 0)
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
                            # val_bbox = val_batch['bbox'].to(weight_dtype).to(
                            #     accelerator.device
                            # )

                            val_controlnet_image, val_sparse_optical_flow, \
                                val_mask, unimatch_flow, val_pixel_values_384, \
                                    val_sparse_optical_flow_384, val_mask_384 = sample_inputs(unimatch, val_pixel_values)
                            

                            fb, fl, fc, fh, fw = val_sparse_optical_flow.shape

                            val_controlnet_flow, _ = get_cmp_flow(
                                cmp, 
                                val_pixel_values_384[:, 0:1, :, :, :].repeat(1, fl, 1, 1, 1), 
                                val_sparse_optical_flow_384, 
                                val_mask_384
                            )
                            # val_controlnet_masks = get_sam_flows(predictor, val_pixel_values)

                            # Inside your training loop
                            val_masks_org = batch['mask']  # Shape: [B, 14, 256, 256]

                            # 1. Resize to match DSI spatial resolution (e.g., 64x64)
                            dsi_mask_spatial = F.interpolate(val_masks_org, size=(64, 64), mode='nearest')

                            # 2. If the model expects 256 channels (SAM feature size), 
                            # you can pad or project. A common trick is to repeat classes:
                            val_controlnet_masks = F.pad(dsi_mask_spatial, (0, 0, 0, 0, 0, 256 - args.num_frames)).to(torch.float16).to(accelerator.device) # Pad 14 to 256





                            val_controlnet_depth, depth_img = get_dav2_flows(depth_anything,val_pixel_values)
                            if fh != 384 or fw != 384:
                                scales = [fh / 384, fw / 384]
                                val_controlnet_flow = F.interpolate(val_controlnet_flow.flatten(0, 1), (fh, fw), mode='nearest').reshape(fb, fl, 2, fh, fw)
                                val_controlnet_flow[:, :, 0] *= scales[1]
                                val_controlnet_flow[:, :, 1] *= scales[0]

                            val_controlnet_image = val_controlnet_image.unsqueeze(1).repeat(1, pixel_values.shape[1], 1, 1, 1)

                            pil_val_pixel_values = [Image.fromarray((val_pixel_values[0][i].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)) for i in range(val_pixel_values.shape[1])]
                            # pil_val_depth = val_controlnet_depth.unsqueeze(1).repeat(1, pixel_values.shape[1], 1, 1, 1)
                            pil_val_depth = np.repeat(depth_img, val_pixel_values.shape[1], axis=1)
                            # pil_val_masks = val_controlnet_masks.unsqueeze(1).repeat(1, pixel_values.shape[1], 1, 1, 1)
                            num_frames = args.num_frames
                            val_output = pipeline(
                                pil_val_pixel_values[0], 
                                pil_val_pixel_values[0],
                                controlnet_flow=val_controlnet_flow,
                                controlnet_depth=val_controlnet_depth,
                                controlnet_mask=val_controlnet_masks,
                                height=args.height,
                                width=args.width,
                                num_frames=num_frames,
                                decode_chunk_size=8,
                                motion_bucket_id=127,
                                fps=7,
                                noise_aug_strength=0.02,
                                # generator=generator,
                            )

                            masks_nps = grayscale_to_color_mask(val_masks_org[0].cpu().numpy(), "cadis")




                            video_frames = val_output.frames[0]

                            for i in range(num_frames):
                                img = video_frames[i]
                                video_frames[i] = np.array(img)
                            video_frames = np.array(video_frames)
                            
                            viz_sparse_flows = []
                            for i in range(val_sparse_optical_flow.shape[1]):
                                temp_flow = val_sparse_optical_flow[0][i].permute(1, 2, 0)
                                viz_sparse_flows.append(flow_to_image(temp_flow))
                            viz_sparse_flows = [np.uint8(np.ones_like(viz_sparse_flows[-1]) * 255)] + viz_sparse_flows
                            viz_sparse_flows = np.stack(viz_sparse_flows)  # [t-1, h, w, c]

                            viz_esti_flows = []
                            for i in range(val_controlnet_flow.shape[1]):
                                temp_flow = val_controlnet_flow[0][i].permute(1, 2, 0)
                                viz_esti_flows.append(flow_to_image(temp_flow))
                            viz_esti_flows = [np.uint8(np.ones_like(viz_esti_flows[-1]) * 255)] + viz_esti_flows
                            viz_esti_flows = np.stack(viz_esti_flows)  # [t-1, h, w, c]

                            viz_unimatch_flow = []
                            for i in range(unimatch_flow.shape[1]):
                                temp_flow = unimatch_flow[0][i].permute(1, 2, 0)
                                viz_unimatch_flow.append(flow_to_image(temp_flow))
                            viz_unimatch_flow = [np.uint8(np.ones_like(viz_unimatch_flow[-1]) * 255)] + viz_unimatch_flow
                            viz_unimatch_flow = np.stack(viz_unimatch_flow)  # [t-1, h, w, c]
                            
                            out_nps = video_frames
                            depth_nps = (pil_val_depth[0]).astype(np.uint8)

                            gt_nps = (val_pixel_values[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
                            ctrl_nps = (val_controlnet_image[0].permute(0, 2, 3, 1).cpu().numpy()*255).astype(np.uint8)
                            sparse_flow_nps = viz_sparse_flows
                            esti_flow_nps = viz_esti_flows
                            unimatch_flow_nps = viz_unimatch_flow

                            out_nps_overlayed = overlay_segmentation(np.array(out_nps), masks_nps)
                            gt_nps_overlayed = overlay_segmentation(np.array(gt_nps), masks_nps)


                            total_nps = np.concatenate([ctrl_nps, depth_nps, unimatch_flow_nps, sparse_flow_nps, 
                                                        esti_flow_nps, out_nps, out_nps_overlayed, gt_nps, gt_nps_overlayed], axis=2)
                            
                            video_name = val_batch['video_name'][0].replace('/', '_').split('.')[0]
                            total_path = os.path.join(val_save_dir,
                                f"step_{global_step}_val_img/{str(val_img_idx).zfill(3)}-{video_name}.mp4",
                            )
                            os.makedirs(os.path.dirname(total_path), exist_ok=True)
                            # torchvision.io.write_video(total_path, total_nps, fps=8, video_codec='h264', options={'crf': '10'})

                            imageio.mimsave(
                                total_path,
                                total_nps,
                                fps=7,
                                quality=10
                            )
                            


                    torch.cuda.empty_cache()
                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_controlnet.restore(controlnet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()



    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        if args.use_ema:
            ema_controlnet.copy_to(controlnet.parameters())

        pipeline =DualFlowControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()
