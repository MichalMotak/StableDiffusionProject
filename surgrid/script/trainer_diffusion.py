import argparse
import yaml
from types import SimpleNamespace

from surgrid.diffusion.denoising_diffusion import Trainer
from surgrid.diffusion.classifier_free_guidance import Unet, GaussianDiffusion
from surgrid.dataset.cadis_dataset import CadisDataset

def dict_to_namespace(d):
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    return d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, help='Path to data config yaml file.')
    args = parser.parse_args()
    
    with open(args.conf, 'r') as f:
        conf = yaml.load(f, yaml.Loader)
    conf = dict_to_namespace(conf)  
    return conf

conf = parse_args()

model = Unet(
    dim = conf.model.dim,
    dim_mults = tuple(map(int, conf.model.dim_mults.split(", "))),
    num_classes = conf.model.num_classes,
    cond_drop_prob = conf.model.cond_drop_prob
)

diffusion = GaussianDiffusion(
    model,
    image_size = conf.dataset.image_size,
    timesteps = conf.diffusion.timesteps,
    sampling_timesteps = conf.diffusion.sampling_timesteps,
    use_cfg_plus_plus = getattr(conf.diffusion, 'use_cfg_plus_plus', False)
)

dataset = CadisDataset(**vars(conf.dataset))

trainer = Trainer(
    diffusion,
    dataset = dataset,
    train_batch_size = conf.trainer.train_batch_size,
    train_lr = float(conf.trainer.train_lr),
    num_workers = conf.trainer.num_workers,
    num_samples = conf.trainer.num_samples,
    save_and_sample_every = conf.trainer.save_and_sample_every,
    results_folder = conf.results_folder,
    train_num_steps = conf.trainer.train_num_steps,
    gradient_accumulate_every = conf.trainer.gradient_accumulate_every,
    ema_decay = conf.trainer.ema_decay,
    amp = True,    
)

if conf.load_checkpoint != "null":
    trainer.load(conf.load_checkpoint)
if conf.dataset.mode == "train":
    trainer.train()
