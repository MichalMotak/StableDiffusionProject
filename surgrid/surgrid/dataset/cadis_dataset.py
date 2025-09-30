import os
import cv2
import glob
import numpy as np
from PIL import Image
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
from surgrid.graph.graph_masked_segclip import GraphEncoder
from surgrid.dataset.cadis_experiments import EXP2
from surgrid.dataset.cadis_utils import remap_mask

class CadisDataset(Dataset):
    def __init__(self,
                 mode,
                 image_size,
                 data_root,
                 split_files,
                 train_graph_encoder=False,
                 return_graph_emb=False,
                 **kwargs
                 ):
        self.mode = mode
        self.image_size = image_size
        self.data_root = data_root
        self.split_files = vars(split_files) if isinstance(split_files, SimpleNamespace) else split_files
        self.train_graph_encoder = train_graph_encoder
        self.return_graph_emb = return_graph_emb

        with open(self.split_files[mode], "r") as f:
            img_file_list = f.read().splitlines()

        if self.mode == 'train':          
            self.diffusion_transform = transforms.Compose([
                                       transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                                       transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0),
                                       transforms.ToTensor()])
            self.graph_transforms = A.Compose([
                                    A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
                                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
        elif mode in ["val", "test"]:
            self.diffusion_transform = transforms.Compose([
                                       transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
                                       transforms.ToTensor()])
            self.graph_transforms = A.Compose([
                                    A.Resize(image_size, image_size, interpolation=cv2.INTER_LANCZOS4),
                                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_path_ = [item for sublist in [glob.glob(os.path.join(self.data_root, l, "Images", "*.png")) for l in img_file_list] for item in sublist]
        segmentation_path_ = [os.path.join(os.path.dirname(os.path.dirname(path)), "Labels", os.path.basename(path)) for path in image_path_]
        scene_graph_path_ = [os.path.join(os.path.dirname(os.path.dirname(path)), "Graphs_EXP2", os.path.splitext(os.path.basename(path))[0] + "_sg.pt") for path in image_path_]
        self._length = len(image_path_)
        self.labels = {
            "image_path_": image_path_,
            "segmentation_path_": segmentation_path_,
            "scene_graph_path_": scene_graph_path_
        }

        if self.return_graph_emb:
            self.embedding_type = kwargs['embedding_type']
            if 'model_masked' in kwargs:
                self.model_masked = GraphEncoder(kwargs["graph_input_dim"], kwargs["graph_hidden_dim"], 
                                                kwargs["graph_embedding_dim"], kwargs["trainable"],
                                                graph_encoder_ckpt = kwargs['model_masked'])
                self.model_masked.eval()
            if 'model_segclip' in kwargs:
                self.model_segclip = GraphEncoder(kwargs["graph_input_dim"], kwargs["graph_hidden_dim"], 
                                                kwargs["graph_embedding_dim"], kwargs["trainable"],
                                                graph_encoder_ckpt = kwargs['model_segclip'])
                self.model_segclip.eval()
            if 'validation_graph' in kwargs:
                # Store embedding for validation graphs
                self.val_graph_emb = [self.get_embedding(torch.load(path), self.embedding_type).detach().squeeze() for path in kwargs['validation_graph']]
                self.val_graph_emb = torch.stack(self.val_graph_emb)
        
    def get_embedding(self, scene_graph, embedding_type):
        if embedding_type == 'masked':
            graph_embeddings = self.model_masked(scene_graph)
        elif embedding_type == 'segclip': 
            graph_embeddings = self.model_segclip(scene_graph)
        elif embedding_type == 'combined':
            graph_embeddings_masked = self.model_masked(scene_graph)
            graph_embeddings_segclip = self.model_segclip(scene_graph)
            graph_embeddings = torch.cat((graph_embeddings_masked, graph_embeddings_segclip), dim=-1)
        return graph_embeddings

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        items = {}

        if self.train_graph_encoder:         
            image = cv2.imread(example["image_path_"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmentation = np.array(Image.open(example["segmentation_path_"]))
            image, segmentation = self.graph_transforms(image=image, mask=segmentation).values()

            items["image"] = torch.tensor(image).permute(2, 0, 1).float()
            items["image_name"] = example["image_path_"]
        
            segmentation = remap_mask(mask=torch.from_numpy(segmentation), exp_dict=EXP2)
            segmentation[segmentation == 255] = 17
            segmentation_unique_class = torch.zeros(18)
            segmentation_unique_class[segmentation.unique().long()] = 1
            items["segmentation_unique_class"] = segmentation_unique_class

            segmentation = np.eye(18)[segmentation]
            segmentation = torch.tensor(segmentation.transpose(2,0,1)).to(dtype=torch.float32)
            items["segmentation"] = segmentation
            
            items["scene_graph"] = torch.load(example["scene_graph_path_"])
            return items

        else:
            #image
            image = Image.open(example["image_path_"])
            image = self.diffusion_transform(image)

            #scene graph embedding
            scene_graph = torch.load(example["scene_graph_path_"])
            graph_embeddings = self.get_embedding(scene_graph, self.embedding_type)
            # graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)
            return image, graph_embeddings.detach().squeeze()