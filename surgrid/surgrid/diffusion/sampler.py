import yaml
import torch
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

class Sampler():
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            conf = yaml.load(f, yaml.Loader)
        conf = dict_to_namespace(conf)
        self.conf = conf

        self.model = Unet(
            dim = conf.model.dim,
            dim_mults = tuple(map(int, conf.model.dim_mults.split(", "))),
            num_classes = conf.model.num_classes,
            cond_drop_prob = conf.model.cond_drop_prob
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = conf.dataset.image_size,
            timesteps = conf.diffusion.timesteps,
            sampling_timesteps = conf.diffusion.sampling_timesteps,
            use_cfg_plus_plus = getattr(conf.diffusion, 'use_cfg_plus_plus', False)
        )

        self.dataset = CadisDataset(**vars(conf.dataset))

        self.trainer = Trainer(
            self.diffusion,
            dataset = self.dataset,
            num_workers = conf.trainer.num_workers,
            results_folder = conf.ckpt_folder,
            amp = True 
        )

        if conf.load_checkpoint != "null":
            self.trainer.load(conf.load_checkpoint)

    def scenegraph_to_image(self, scene_graph, cond_scale = 2.5, batch_size = 1):
        graph_embeddings = self.dataset.get_embedding(scene_graph, self.conf.dataset.embedding_type)
        graph_embeddings = graph_embeddings.repeat(batch_size, 1)

        graph_embeddings = graph_embeddings.to(dtype=torch.float32, device=self.conf.device)
        image = self.diffusion.sample(classes = graph_embeddings, cond_scale = cond_scale)
        return image
