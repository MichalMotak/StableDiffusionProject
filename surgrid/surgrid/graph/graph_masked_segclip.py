
import yaml
import torch
import random
from torch import nn
from omegaconf import OmegaConf
import torch.nn.functional as F

from surgrid.taming.taming.models.vqgan import VQModel
from surgrid.graph.graph_encoder import GCNGraphEncoder
from surgrid.graph.mm_transformer_module import BasicTransformerBlock, get_linear_feas_by_hook

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss

def choose_random_class(unique_class, device):
    """  
    # We are focusing only on tools and iris 
    # Bounding box of iris should also cover pupil. To learn pupil and iris sizes
    # From all the unique class found in image, remove the ignore class and other anatomy class
    # TODO: CADIS related specifications. For other dataset, you can do the same, or remove it.
    """
    desired_class = [4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    desired_class = torch.tensor(desired_class, device=device)
    filtered_class = unique_class[torch.isin(unique_class, desired_class)]

    iris_class = filtered_class[filtered_class == 4]
    tool_class = filtered_class[filtered_class > 4]

    # Check if both are non-empty first
    if iris_class.numel() == 0:
        return tool_class[torch.randint(len(tool_class), (1,)).item()]
    if tool_class.numel() == 0:
        return iris_class[0]
    
    # Only select iris 10% of the time, priotizing tools 
    # Considering size of bbox, and bigger loss, also frames without tools
    if random.random() < 0.10:
        return iris_class[0]
    else:
        return tool_class[torch.randint(len(tool_class), (1,)).item()]

def mask_to_bbox(binary_mask):
    non_zero_coords = torch.nonzero(binary_mask.squeeze(), as_tuple=False)

    # Find the min and max values for x and y coordinates
    ymin, xmin = torch.min(non_zero_coords, dim=0).values
    ymax, xmax = torch.max(non_zero_coords, dim=0).values
    
    # Create a bounding box (xmin, ymin, xmax, ymax)
    bbox = (xmin.item(), ymin.item(), xmax.item(), ymax.item())
    return bbox

def random_object_masked_image(batch):
    masks, bbox_masks = [], []

    B, D, H, W = batch["image"].size()
    segmentation_unique_class = batch["segmentation_unique_class"]
    # To ensure that ignore class is not reconstructed
    segmentation_unique_class[:, -1] = 0

    # Selecting random existing object in segmentation map to create masked image
    for i in range(B):
        unique_class = torch.nonzero(segmentation_unique_class[i,:]).squeeze()
        # Weighted random class to favour tools more than anatomy
        random_class = choose_random_class(unique_class, device=batch["image"].device)

        # Dilate the mask (Dilate primary knive even more)
        mask = batch["segmentation"][i,random_class,:,:]
        if random_class.item() == 10: kernel = 27
        else: kernel = 21
        padding = kernel // 2

        mask = F.conv2d(mask.unsqueeze(0).unsqueeze(0), torch.ones((1, 1, kernel, kernel), dtype=mask.dtype, device=mask.device), padding=padding)
        mask = (mask > 0).float().squeeze(0)
        
        bbox = mask_to_bbox(mask)
        bbox_mask = torch.zeros(1, H, W, dtype=mask.dtype, device=mask.device)
        bbox_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        mask = mask.expand(D,H,W)
        masks.append(1 - mask)
        bbox_mask = bbox_mask.expand(D,H,W)
        bbox_masks.append(1 - bbox_mask)

    masks = torch.stack(masks)
    masked_image = batch["image"] * masks
    return masked_image
    

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
    

class SegmentationEncoder(nn.Module):
    """
    Encode segmentation to a fixed size vector
    """
    def __init__(
        self,
        device,
        segmentation_encoder_config,
        segmentation_encoder_ckpt
    ):
        super().__init__()
        self.config = self.load_config(segmentation_encoder_config)
        self.model = self.load_vqgan(self.config, segmentation_encoder_ckpt).to(device)
        
    def forward(self, x):
        z, _, [_, _, indices] = self.model.encode(x)
        encodings = z.squeeze(1).view(z.size(0), -1)
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings
    
    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None):
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        if sd is not None:  
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()
    

class ImageEncoder(nn.Module):
    """
    Encode image to a fixed size vector
    """
    def __init__(
        self,
        device,
        image_encoder_config,
        image_encoder_ckpt
    ):
        super().__init__()
        self.config = self.load_config(image_encoder_config)
        self.model = self.load_vqgan(self.config, image_encoder_ckpt).to(device)
        
    def forward(self, x):
        z, _, [_, _, indices] = self.model.encode(x)
        encodings = z.squeeze(1).view(z.size(0), -1)
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings
    
    def load_config(self, config_path, display=False):
        config = OmegaConf.load(config_path)
        if display:
            print(yaml.dump(OmegaConf.to_container(config)))
        return config

    def load_vqgan(self, config, ckpt_path=None):
        model = VQModel(**config.model.params)
        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        if sd is not None:  
            missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.eval()


class GraphEncoder(nn.Module):
    """
    Encode graphs to a fixed size vector
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        z_dim, 
        trainable,
        device="cpu",
        dropout=0.5,
        graph_encoder_ckpt=None
    ):
        super().__init__()
        self.model = GCNGraphEncoder(input_dim, hidden_dim, z_dim, dropout, global_pooling=True)
        if graph_encoder_ckpt is not None:
            graph_encoder_weights = torch.load(graph_encoder_ckpt, map_location=device)
            graph_encoder_weights = {k.replace("graph_encoder.model.", ""): v for k, v in graph_encoder_weights.items() if k.startswith("graph_encoder")}
            self.model.load_state_dict(graph_encoder_weights)
            self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, graph):
        # graph.x = normalize(graph.x, 0.0, 1.0).to(torch.float32)
        # graph.x[:, -4:] = (graph.x[:, -4:] + 1.0) / 2.0
        graph.x = graph.x.to(torch.float32)
        encodings = self.model(graph.x, graph.edge_index, graph.batch) 
        # encodings = F.normalize(encodings, p=2, dim=-1)
        return encodings


class SegClipModel(nn.Module):
    def __init__(
        self,
        temperature,
        segmentation_dim,
        segmentation_encoder,
        graph_encoder
    ):
        super().__init__()
        self.temperature = temperature
        self.segmentation_dim = segmentation_dim
        self.segmentation_encoder = segmentation_encoder
        self.graph_encoder = graph_encoder

    def forward(self, batch):
        # Getting Image and Graph Features
        segmentation_embeddings = self.segmentation_encoder(batch["segmentation"])
        graph_embeddings = self.graph_encoder(batch["scene_graph"])

        # Calculating the Loss
        logits = (graph_embeddings @ segmentation_embeddings.T) / self.temperature
        segmentation_similarity = segmentation_embeddings @ segmentation_embeddings.T
        
        targets = F.softmax(
            (segmentation_similarity) / self.temperature, dim=-1
        )
        graphs_loss = cross_entropy(logits, targets)
        segmentation_loss = cross_entropy(logits.T, targets.T)
        loss =  (segmentation_loss + graphs_loss) / 2.0 
        return loss.mean()
    

class MaskedLocalModel(nn.Module):
    def __init__(
        self,
        dropout,
        image_embedding_dim,
        image_encoder,
        graph_embedding_dim,
        graph_encoder
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.graph_encoder = graph_encoder
        self.basic_transformer = BasicTransformerBlock(dim=image_embedding_dim, n_heads=8, d_head=64, dropout=dropout, context_dim=graph_embedding_dim)
        self.criterion = nn.MSELoss()
    
    def forward(self, batch):
        # Getting Image and Graph Features
        graph_embeddings = self.graph_encoder(batch["scene_graph"])
        masked_image = random_object_masked_image(batch)
        
        image_gt_embeddings = self.image_encoder(batch["image"]).unsqueeze(1).contiguous()
        image_masked_embeddings = self.image_encoder(masked_image).unsqueeze(1).contiguous()
        image_reconstructed_embeddings = self.basic_transformer(image_masked_embeddings, context=graph_embeddings.unsqueeze(1))
        
        # Calculating the Loss
        loss = self.criterion(image_reconstructed_embeddings, image_gt_embeddings)
        return loss