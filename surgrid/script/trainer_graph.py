import gc
import argparse
import datetime
import os
import logging
import torch
from torch_geometric.loader import DataLoader
from tqdm.autonotebook import tqdm
from omegaconf import OmegaConf

from surgrid.dataset.cadis_dataset import CadisDataset
from surgrid.graph.graph_masked_segclip import *

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def create_logger(log_file_path):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG) 

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def build_loaders(size, mode, data_root, split_files, batch_size, num_workers):
    dataset = CadisDataset(
        mode=mode,
        image_size=size,
        data_root=data_root,
        split_files=split_files,
        train_graph_encoder=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "image_name"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "image_name"}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def train(mode,      
          size=128,
          batch_size=32,
          num_workers=8,
          weight_decay=1e-5,
          patience=1,
          factor=0.8,
          epochs=200,
      
          graph_encoder_lr_segclip=1e-6,
          graph_encoder_lr_masked=1e-6, 
          graph_input_dim=21,
          graph_hidden_dim=256,
          graph_embedding_dim=256,
          graph_encoder_ckpt=None,

          image_encoder_lr=1e-6,
          image_embedding_dim=256,
          image_encoder_config=None,
          image_encoder_ckpt=None,

          segmentation_embedding_dim=256,
          segmentation_encoder_config=None,
          segmentation_encoder_ckpt=None,

          trainable=True,
          temperature=1.0,
          dropout=0.25,
          data_root=None,
          split_files=None,
          log_dir=None,
          device="cuda"):

    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    if slurm_job_id is not None:
        exp_dir = f"{log_dir}/graphencoder_{mode}_{slurm_job_id}-{timestamp}"
    else:
        exp_dir = f"{log_dir}/graphencoder_{mode}-{timestamp}"

    checkpoint_dir = f"{exp_dir}/checkpoints"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    log = create_logger(f"{exp_dir}/logfile.log")
    log.info(f"{torch.cuda.is_available()=}")

    train_loader = build_loaders(size=size, mode="train", data_root=data_root, split_files=split_files, batch_size=batch_size, num_workers=num_workers)
    valid_loader = build_loaders(size=size, mode="val", data_root=data_root, split_files=split_files, batch_size=batch_size, num_workers=num_workers)

    if mode == "segclip":
        graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_embedding_dim, trainable)
        segmentation_encoder = SegmentationEncoder(device, segmentation_encoder_config, segmentation_encoder_ckpt)
        model = SegClipModel(temperature, segmentation_embedding_dim, segmentation_encoder, graph_encoder).to(device)
        params = [{"params": model.graph_encoder.parameters(), "lr": graph_encoder_lr_segclip},
                 ]
        
    elif mode == "masked":
        graph_encoder = GraphEncoder(graph_input_dim, graph_hidden_dim, graph_embedding_dim, trainable)
        image_encoder = ImageEncoder(device, image_encoder_config, image_encoder_ckpt)
        model = MaskedLocalModel(dropout, image_embedding_dim, image_encoder, graph_embedding_dim, graph_encoder).to(device)
        params = [{"params": model.basic_transformer.parameters(), "lr": image_encoder_lr},
                  {"params": model.graph_encoder.parameters(), "lr": graph_encoder_lr_masked}
                 ]

    optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=patience, factor=factor)

    best_loss = float('inf')
    for epoch in range(epochs):
        log.info(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, device)

        log.info(f"### Training Loss: {train_loss} ###")
        log.info(f"### Validation Loss: {valid_loss} ###")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'latest_val_loss.pth'))

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            log.info("### New best validation loss ###")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_val_loss.pth'))

        lr_scheduler.step(valid_loss.avg)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["segclip", "masked"])
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()   

    config = OmegaConf.load(args.config)
    train(mode=args.mode, **config)