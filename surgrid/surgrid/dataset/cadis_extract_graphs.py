import os
from glob import glob

import cv2
import torch
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from surgrid.dataset.cadis_utils import remap_mask
from surgrid.dataset.cadis_experiments import EXP1, EXP2, EXP3

def create_graph_from_mask(mask: np.ndarray,
                           num_classes: int,
                           background_label: int = None,
                           max_distance: int = 50,
                           gaussian_blur_kernel_size: int = 5,
                           apply_gaussian_blur: bool = False,
                           morph_kernel_size: int = 2,
                           touch_threshold: int = 3,
                           min_area: int = 10,
                           min_aspect_ratio: float = 0.1) -> nx.Graph:
    """
    Create a graph from a segmentation mask with preprocessing to handle noise.

    :param mask: Segmentation mask as a numpy array.
    :param num_classes: Total number of classes, including background.
    :param background_label: Label of the background class, if any.
    :param max_distance: Maximum distance between connected components to add an edge. TODO: Remove?
    :param gaussian_blur_kernel_size: Kernel size for Gaussian blur.
    :param apply_gaussian_blur: Whether to apply Gaussian blur as a noise reduction step.
    :param morph_kernel_size: Kernel size for morphological operations.
    :param touch_threshold: Minimum overlap required to consider components as touching.
    :param min_area: Minimum area for a component to be considered significant.
    :param min_aspect_ratio: Minimum aspect ratio to consider a component significant.
    :return: A NetworkX graph representing the segmented objects and their relationships.
    """

    # Convert the mask to a numpy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Pre-process mask to reduce impact of annotation noise
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)  # Convert mask to uint8 if it's not already

    if apply_gaussian_blur:
        mask = cv2.GaussianBlur(mask, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    # Erosion followed by dilation to remove small components
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    G = nx.Graph()
    image_height, image_width = mask.shape

    components = {}

    # First pass: identify components and add nodes
    for class_id in range(0, num_classes):  # Assuming class IDs start from 0
        if class_id == background_label:
            continue

        # Create a binary mask for the current class
        class_mask = (mask == class_id).astype(np.uint8)

        # Find connected components (instances) within the class
        num_labels, labels_im = cv2.connectedComponents(class_mask)

        for i in range(1, num_labels):  # TODO: Start from 1 to ignore the background label?

            component_mask = (labels_im == i).astype(np.uint8)

            # Calculate component properties
            ys, xs = np.where(component_mask)
            if len(xs) == 0 or len(ys) == 0:  # Skip empty components
                continue
            area = len(xs)
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            aspect_ratio = (y_max - y_min + 1) / (x_max - x_min + 1)

            if area < min_area or aspect_ratio < min_aspect_ratio:
                continue  # Skip components that don't meet the criteria

            relative_width = (x_max - x_min) / image_width
            relative_height = (y_max - y_min) / image_height
            relative_centroid_x = xs.mean() / image_width
            relative_centroid_y = ys.mean() / image_height

            one_hot_class = [0] * num_classes
            one_hot_class[class_id] = 1
            features = one_hot_class + [relative_width, relative_height, relative_centroid_x, relative_centroid_y]

            # Add node to the graph
            node_id = len(G.nodes)  # Unique ID for the node
            G.add_node(node_id, features=features, centroid=(relative_centroid_x, relative_centroid_y))
            components[node_id] = component_mask

    # Second pass: add edges between touching components
    for idx1, data1 in G.nodes(data=True):

        for idx2, data2 in G.nodes(data=True):
            if idx1 >= idx2:
                continue

            # Use individual component masks for each node
            component1_mask = components[idx1]
            component2_mask = components[idx2]

            # Check if the dilated components are touching
            dilated_component1 = cv2.dilate(component1_mask, np.ones((3, 3), np.uint8), iterations=1)
            touching = cv2.bitwise_and(dilated_component1, component2_mask)

            # Add an edge if components are touching
            if np.sum(touching) >= touch_threshold:
                G.add_edge(idx1, idx2)

        data1['x'] = data1['features']

    return G


def process_mask_file(mask_file: str):
    mask_file_name = mask_file.split("/")[-1]
    mask_sample_path = mask_file.removesuffix(mask_file_name)

    sg_sample_path_exp1 = mask_sample_path.replace("Labels", "Graphs_EXP1")
    os.makedirs(sg_sample_path_exp1, exist_ok=True)
    sg_sample_path_exp2 = mask_sample_path.replace("Labels", "Graphs_EXP2")
    os.makedirs(sg_sample_path_exp2, exist_ok=True)
    sg_sample_path_exp3 = mask_sample_path.replace("Labels", "Graphs_EXP3")
    os.makedirs(sg_sample_path_exp3, exist_ok=True)

    target_sg_file_name = mask_file_name.split(".")[0] + "_sg.pt"
    target_sg_file_path_exp1 = os.path.join(sg_sample_path_exp1, target_sg_file_name)
    target_sg_file_path_exp2 = os.path.join(sg_sample_path_exp2, target_sg_file_name)
    target_sg_file_path_exp3 = os.path.join(sg_sample_path_exp3, target_sg_file_name)

    mask = cv2.imread(mask_file, 0)  # Load as grayscale
    # Map masks to different experiment settings
    mask = torch.from_numpy(mask)
    mask_exp1 = remap_mask(mask, EXP1).numpy()
    mask_exp2 = remap_mask(mask, EXP2).numpy()
    mask_exp3 = remap_mask(mask, EXP3).numpy()

    scene_graph_exp1 = create_graph_from_mask(mask_exp1,
                                              num_classes=9 - 1,   # w/o ignore label
                                              background_label=None,
                                              morph_kernel_size=5,
                                              min_area=50,
                                              min_aspect_ratio=0.1)
    pyg_exp1 = from_networkx(scene_graph_exp1)
    if pyg_exp1.x is None:
        raise ValueError("Node feature matrix x is not set.")

    scene_graph_exp2 = create_graph_from_mask(mask_exp2,
                                              num_classes=18 - 1,
                                              background_label=None,
                                              morph_kernel_size=5,
                                              min_area=50,
                                              min_aspect_ratio=0.1)
    pyg_exp2 = from_networkx(scene_graph_exp2)
    if pyg_exp2.x is None:
        raise ValueError("Node feature matrix x is not set.")

    scene_graph_exp3 = create_graph_from_mask(mask_exp3,
                                              num_classes=26 - 1,
                                              background_label=None,
                                              morph_kernel_size=5,
                                              min_area=50,
                                              min_aspect_ratio=0.1)
    pyg_exp3 = from_networkx(scene_graph_exp3)
    if pyg_exp3.x is None:
        raise ValueError("Node feature matrix x is not set.")

    torch.save(pyg_exp1, target_sg_file_path_exp1)
    torch.save(pyg_exp2, target_sg_file_path_exp2)
    torch.save(pyg_exp3, target_sg_file_path_exp3)
    return scene_graph_exp2


if __name__ == "__main__":

    root = '/path_to_dataset/CaDISv2/'
    mask_files = glob(os.path.join(root, f"Video*/Labels/*.png"))

    Parallel(n_jobs=12)(delayed(process_mask_file)(mask_file) \
                        for mask_file in tqdm(mask_files,
                                              total=len(mask_files),
                                              desc="Converting CaDISv2 masks to SGs"))