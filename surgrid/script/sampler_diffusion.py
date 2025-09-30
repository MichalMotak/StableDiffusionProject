import os
import torch
import argparse
from torchvision import utils
from surgrid.diffusion.sampler import Sampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, help='Path to data config yaml file.')
    args = parser.parse_args()
    return args

args = parse_args()
surgrid = Sampler(config_path=args.conf)
os.makedirs(surgrid.conf.save_folder, exist_ok=True)

scene_graph_path = "/home/mmotaksano/SurGrID/dataset/CaDISv2/Video01/Graphs_EXP1/Video1_frame000090_sg.pt"
save_path = os.path.join(surgrid.conf.save_folder, os.path.basename(scene_graph_path).replace(".pt", ".png"))
scene_graph = torch.load(scene_graph_path)
image = surgrid.scenegraph_to_image(scene_graph)
utils.save_image(image, save_path)
