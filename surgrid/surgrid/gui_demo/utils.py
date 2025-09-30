import torch
from torch_geometric.data import Data

from surgrid.dataset.cadis_experiments import EXP1, EXP2, EXP3
from surgrid.dataset.cadis_utils import get_cadis_float_cmap


def node_ft_to_class(node_ft: torch.Tensor, num_classes: int):
    return torch.argmax(node_ft[:num_classes]).item()


def data_to_colors_cadisv2(data: Data, exp_id: int):
    if exp_id == 1:
        num_classes = 9
    elif exp_id == 2:
        num_classes = 18
    elif exp_id == 3:
        num_classes = 26
    else:
        raise ValueError(f"Invalid exp_id: {exp_id}")

    cmap = get_cadis_float_cmap().to(data.x.device)

    return [cmap[node_ft_to_class(data.x[n], num_classes - 1)] for n in range(data.num_nodes)]

def rgb_to_hex(rgb: tuple) -> str:
    """
    Convert an RGB color tuple to a hexadecimal color string.

    :param rgb: The RGB color tuple.
    :return: The hexadecimal color string.
    """
    if max(rgb) > 1:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    return '#{:02x}{:02x}{:02x}'.format(*[round(x * 255) for x in rgb])


def gt_filepath_to_img_path_cadisv2(gt_filepath: str) -> str:
    """
    Convert a ground truth CaDisv2 graph file path to an image file path.

    :param gt_filepath: The ground truth file path.
    :return: The image file path.
    """
    exp = gt_filepath.split("/")[-2]
    return gt_filepath.replace(f"/{exp}/", f"/Images/").replace("_sg.pt", ".png")

