import os
import cv2
import glob
import numpy as np
import albumentations as A
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from surgrid.dataset.cadis_experiments import EXP2
from surgrid.dataset.cadis_utils import remap_mask

class CadisBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=256,
                 augment=False,
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.augment = augment

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        
        image_path_ = [item for sublist in [glob.glob(os.path.join(self.data_root, l, "Images", "*.png")) for l in self.image_paths] for item in sublist]
        segmentation_path_ = [os.path.join(os.path.dirname(os.path.dirname(path)), "Labels", os.path.basename(path)) for path in image_path_]
       
        self.size = size
        if self.augment:
            self.rescaler = A.Compose([A.OneOf([
                                    A.Resize(height=self.size, width=self.size, interpolation=cv2.INTER_LANCZOS4),
                                    A.Compose([A.SmallestMaxSize(max_size=self.size, interpolation=cv2.INTER_LANCZOS4), A.RandomCrop(height=self.size, width=self.size)]),
                                    # To apply random crop also on vertical wise
                                    A.Compose([A.Resize(height=self.size, width=int(self.size*1.75), interpolation=cv2.INTER_LANCZOS4), A.RandomCrop(height=self.size, width=self.size)]),
                                    A.Compose([A.Resize(height=int(self.size*1.5), width=int(self.size*1.5), interpolation=cv2.INTER_LANCZOS4), A.RandomCrop(height=self.size, width=self.size)]),
                                     ], p=1.0)])
        else:
            self.rescaler = A.Resize(height=self.size, width=self.size, interpolation=cv2.INTER_LANCZOS4)
        
        self.augmentation = A.Compose([A.HorizontalFlip(p=0.5),
                                       A.RandomBrightnessContrast(p=0.5),
                                       A.RandomGamma(p=0.5)
                                       ])

        self.labels = {
            "image_path_": image_path_,
            "segmentation_path_": segmentation_path_,
        }
        self._length = len(self.labels["image_path_"])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        ex = {}
        
        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        segmentation = np.array(Image.open(example["segmentation_path_"])).astype(np.uint8)
        img, segmentation = self.rescaler(image=img, mask=segmentation).values()
        
        if self.augment:
            img, segmentation = self.augmentation(image=img, mask=segmentation).values()
        
        # image
        ex["image"] = (img / 127.5 - 1.0).astype(np.float32)

        # segmentation cadis
        segmentation = remap_mask(mask=torch.from_numpy(segmentation), exp_dict=EXP2)
        segmentation[segmentation == 255] = 17
        segmentation = np.eye(18)[segmentation]
        segmentation = torch.tensor(segmentation).to(dtype=torch.float32)
        ex["segmentation"] = segmentation

        return ex