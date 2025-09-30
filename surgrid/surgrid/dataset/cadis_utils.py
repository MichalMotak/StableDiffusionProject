import torch
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

def remap_mask(mask: torch.Tensor, exp_dict: dict, ignore_label: int = 255):
    classes = []
    class_remapping = exp_dict["LABEL"]
    for key, val in class_remapping.items():
        for cls in val:
            classes.append(cls)
    assert len(classes) == len(set(classes))

    N = max(len(classes), mask.max() + 1)
    remap_array = np.full(N, ignore_label, dtype=np.uint8)
    for key, val in class_remapping.items():
        for v in val:
            remap_array[v] = key
    mask = mask.int()
    remap_mask = remap_array[mask]
    remap_mask_tensor = torch.from_numpy(remap_mask)
    return remap_mask_tensor


def get_cadis_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [0, 137, 255],
            [255, 165, 0],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
            [170, 124, 0],
            [188, 255, 0],
            [0, 207, 255],
            [0, 255, 207],
            [188, 0, 255],
            [243, 0, 255],
            [0, 203, 108],
            [252, 255, 0],
            [93, 182, 177],
            [0, 81, 203],
            [211, 183, 120],
            [231, 203, 0],
            [0, 124, 255],
            [10, 91, 44],
            [2, 0, 60],
            [0, 144, 2],
            [133, 59, 59],
        ]
    )


def get_cadis_float_cmap():
    return torch.from_numpy(get_cadis_colormap())/255.0


def convert_mask_to_RGB(mask: torch.Tensor,
                        palette: torch.Tensor,
                        ignore_index = None,
                        ignore_add_channel: bool = False) -> torch.Tensor:
    """
    Convert a segmentation mask into an RGB image.

    Parameters:
    mask (torch.Tensor): The segmentation mask, shape (B, H, W).
    palette (torch.Tensor): The color palette for segmentation map, shape (num_classes, 3).

    Returns:
    rgb_images (torch.Tensor): The RGB images, shape (B, 3, H, W).
    """

    # Check if ignore index exists in the mask
    if ignore_index is not None and (mask == ignore_index).any():
        # Extend the palette to have an entry for label 255
        palette = torch.cat([palette, torch.tensor([[0, 0, 0]], device=palette.device)], dim=0)
        mask = mask.clone()  # Clone to ensure we don't modify the original mask in place
        mask[mask == ignore_index] = palette.size(0) - 1

    # Convert the mask to one-hot encoded tensor
    mask_onehot = F.one_hot(mask, num_classes=palette.shape[0]).permute(0, 3, 1,
                                                                        2).float()  # shape: (B, num_classes, H, W)
    # mask_onehot.to(palette.device)

    # Expand palette dimensions to match mask_onehot
    palette = palette[None, :, :, None, None]  # shape: (1, num_classes, 3, 1, 1)

    # Convert one-hot to rgb by multiplying with palette and summing over the classes dimension
    rgb_images = (palette * mask_onehot[:, :, None, :, :]).sum(dim=1)  # shape: (B, 3, H, W)

    return rgb_images


def visualize_data(image,
                   mask,
                   graph,
                   num_classes: int,
                   class_names,
                   palette = None,
                   ignore_index = None,
                   save_path: str = None):
    """
    Visualizes an image, a segmentation mask and its corresponding graph side by side,
    including edges between nodes and class names. At least one of image, mask or graph must be provided.
    # TODO: Fix case if no mask provided

    :param image: The image as a PyTorch tensor in [C, H, W].
    :param mask: The segmentation mask as a numpy array or a PyTorch tensor in [H, W].
    :param graph: The corresponding graph as a NetworkX graph.
    :param num_classes: Total number of classes (including background if any).
    :param class_names: A list of strings representing the names of the classes.
    :param palette: Optional. A color palette. If None, 'viridis' colormap is used.
    :param ignore_index: Optional. Ignore index for the mask. If None, no ignore index is used.
    :param save_path: Optional. Path to save plot to. If None, plot will not be saved and only be rendered.
    """

    assert not (image is None and mask is None and graph is None)

    # Convert image to a numpy array if it's a tensor
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Convert mask to a numpy array if it's a tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Calculate aspect ratio based on mask dimensions
    if isinstance(mask, torch.Tensor):
        mask_height, mask_width = mask.shape[-2:]
    else:
        mask_height, mask_width = mask.shape

    n_subplots = (image is not None) + (mask is not None) + (graph is not None)
    aspect_ratio = mask_width / mask_height
    inverse_aspect_ratio = mask_height / mask_width
    subplot_width = 15 / n_subplots
    subplot_height = subplot_width / aspect_ratio
    plt.figure(figsize=(15, subplot_height))

    plot_count = 0

    #
    #   Visualize the image
    #

    if image is not None:
        plot_count += 1
        plt.subplot(1, n_subplots, plot_count)
        plt.title('Image')
        plt.imshow(image.transpose(1, 2, 0))

    #
    #   Visualize the mask
    #

    if mask is not None:
        plot_count += 1
        plt.subplot(1, n_subplots, plot_count)
        plt.title('Segmentation Mask')
        rgb_mask = convert_mask_to_RGB(torch.from_numpy(mask).unsqueeze(0), palette=palette, ignore_index=ignore_index)
        plt.imshow(rgb_mask.squeeze(0).permute(1, 2, 0).cpu().numpy())

    #
    #   Visualize the graph
    #

    if graph is not None:
        plot_count += 1
        ax2 = plt.subplot(1, n_subplots, plot_count)
        plt.title('Graph Representation')
        pos = {node: (data['features'][-2] * mask.shape[1], (1 - data['features'][-1]) * mask.shape[0]) for node, data in
               graph.nodes(data=True)}
        # Create a color map for the nodes based on their class. Default to 'viridis' if no cmap provided.
        if palette is None:
            cmap = 'viridis'
        else:
            cmap = LinearSegmentedColormap.from_list('CustomCMAP', palette.numpy()[:num_classes - 1])
        mask_cmap = plt.cm.get_cmap(cmap, num_classes)
        class_colors = mcolors.ListedColormap(mask_cmap(np.linspace(0, 1 + 1 / num_classes, num_classes)))

        # Add labels to the nodes (e.g., class name and node ID)
        for node, (x, y) in pos.items():
            features = graph.nodes[node]['features']
            class_id = np.argmax(features[:num_classes - 1])  # Extract class ID from one-hot vector  # w/o ignore
            class_name = class_names[class_id]  # Get class name based on class ID
            node_color = class_colors(class_id / num_classes)  # Map class ID to color
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=200, node_color=[node_color])
            plt.text(x, y, f'{class_name}\nID:{node}', fontsize=9, ha='right', va='center')

        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1)

        ax2.set_box_aspect(inverse_aspect_ratio)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    plt.show()


def visualize_graph_list(data_list,
                         image_shape: tuple,
                         num_classes: int,
                         class_names,
                         node_size=200,
                         save_path: str = None,
                         palette=None):
    """
    Visualizes a list of graphs with nodes colored by their class, keeping colors consistent for class IDs.

    Args:
        data_list (list[Data]): A list of PyTorch Geometric Data objects representing the graphs to visualize.
        image_shape (tuple): The shape of the original image (height, width).
        num_classes (int): The number of different classes for coloring nodes.
        class_names (list[str]): A list of class names corresponding to each class.
        save_path (str, optional): Path to save the visualized graph. If None, the graph is not saved.
        palette: The palette to use for coloring nodes based on their class.
    """

    # Create a color map for the nodes based on their class. Default to 'viridis' if no cmap provided.
    if palette is None:
        cmap = 'viridis'
    else:
        cmap = LinearSegmentedColormap.from_list('CustomCMAP', palette.numpy()[:num_classes - 1])  # w/o ignore
    mask_cmap = plt.cm.get_cmap(cmap, num_classes)
    class_colors = mcolors.ListedColormap(mask_cmap(np.linspace(0, 1 + 1 / num_classes, num_classes)))

    num_graphs = len(data_list)
    fig, axes = plt.subplots(1, num_graphs, figsize=(15 * num_graphs, 7), frameon=False)

    for idx, graph_data in enumerate(data_list):

        ax = axes[idx] if num_graphs > 1 else axes

        # Convert to NetworkX graph if necessary
        if isinstance(graph_data, Data):
            graph = to_networkx(graph_data.cpu(), to_undirected=True)
            # Manually transfer node features from PyG Data to NetworkX
            for node, data in enumerate(graph_data.x.cpu()):
                graph.nodes[node]['features'] = data
        else:
            graph = graph_data

        try:
            pos = {node: (data['features'][-2] * image_shape[1], (1 - data['features'][-1]) * image_shape[0]) for node, data in
                   graph.nodes(data=True)}
        except KeyError:
            pos = {node: (data['x'][-2] * image_shape[1], (1 - data['x'][-1]) * image_shape[0]) for
                   node, data in
                   graph.nodes(data=True)}

        # Add labels to the nodes (e.g., class name and node ID)
        for node, (x, y) in pos.items():
            features = graph.nodes[node]['features']
            class_id = np.argmax(features[:num_classes - 1])  # Extract class ID from one-hot vector  # w/o ignore
            class_name = class_names[class_id]  # Get class name based on class ID
            node_color = class_colors(class_id / num_classes)  # Map class ID to color
            nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_size=node_size, node_color=[node_color], ax=ax)
            ax.text(x, y, f'{class_name}\nID:{node}', fontsize=9, ha='right', va='center')

        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=1, ax=ax)

        ax.set_aspect('equal')
        # ax.set_title(f"Graph {idx + 1}")

    # plt.axis('equal')  # Set the aspect ratio to be equal
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    plt.show()


def convert_torch_masks_to_torch_geometric_data(mask: torch.Tensor, extract_graph_func, num_classes: int) -> Batch:
    data_list = []
    for _m in mask:
        # Convert the mask to a numpy array if it's a tensor
        if isinstance(_m, torch.Tensor):
            _m = _m.cpu().numpy()

        # Create a graph from the mask (networkx object)
        graph = extract_graph_func(_m, num_classes=num_classes)

        # Extract node features and edge index
        node_features = []
        for node, data in graph.nodes(data=True):
            # Extract the feature vector from node attributes
            feature_vector = data['features']
            node_features.append(feature_vector)

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    # Use PyTorch Geometric's Batch class to batch the data objects
    batched_data = Batch.from_data_list(data_list)

    return batched_data


def adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """
    Converts an adjacency matrix to an edge_index tensor suitable for PyTorch Geometric,
    excluding self-edges.

    Args:
        adj (torch.Tensor): The adjacency matrix [num_nodes, num_nodes].

    Returns:
        torch.Tensor: The edge_index tensor [2, num_edges].
    """
    # Set the diagonal elements to zero to exclude self-edges
    adj.fill_diagonal_(0)

    # Apply a threshold to get binary edges and exclude self-edges
    edge_index = (adj > 0.5).nonzero(as_tuple=False).t().contiguous()
    return edge_index


def predicted_tensor_to_data_objects(recon_features: torch.Tensor,
                                     recon_adj,
                                     batch_vector: torch.Tensor):
    """
    Converts tensors of reconstructed node features and adjacency matrices into a list of PyTorch Geometric Data objects.

    Args:
        recon_features (torch.Tensor): The reconstructed node features [batch_size, num_nodes, node_feature_dim].
        recon_adj (List[torch.Tensor]): A list of reconstructed adjacency matrices, one for each graph in the batch.
        batch_vector (torch.Tensor): The batch vector indicating the graph each node belongs to.

    Returns:
        List[Data]: A list of PyTorch Geometric Data objects, one for each graph in the batch.
    """
    num_graphs = batch_vector.max().item() + 1

    data_list = []

    for i in range(num_graphs):
        # Get the nodes and features corresponding to the current graph
        node_mask = (batch_vector == i)
        graph_features = recon_features[node_mask]

        # Get the reconstructed adjacency matrix for the current graph
        graph_adj = recon_adj[node_mask][:, node_mask]

        # Convert the adjacency matrix to edge_index format if it's not already
        edge_index = adj_to_edge_index(graph_adj)

        # Create a PyTorch Geometric Data object for the graph
        # TODO: Conistent use of x / features!
        data = Data(x=graph_features, features=graph_features, edge_index=edge_index)
        data_list.append(data)

    # TODO: Use batched data instead of list of data?
    return data_list


def separate_adjacency_matrix(decoded_adj: torch.Tensor, batch_vector: torch.Tensor):
    """
    Separates a large adjacency matrix for a minibatch into a list of individual adjacency matrices for each graph.

    Args:
        decoded_adj (torch.Tensor): The large decoded adjacency matrix for the whole minibatch.
        batch_vector (torch.Tensor): The batch vector indicating the graph each node belongs to.

    Returns:
        list[torch.Tensor]: A list containing the individual adjacency matrices for each graph.
    """
    num_graphs = batch_vector.max().item() + 1  # Number of graphs in the batch
    separated_adjs = []

    for i in range(num_graphs):
        # Get the nodes corresponding to the current graph
        mask = (batch_vector == i)

        # Extract the submatrix for the current graph
        sub_adj = decoded_adj[mask][:, mask]
        separated_adjs.append(sub_adj)

    return separated_adjs


def torch_geometric_to_networkx(data: Data) -> nx.Graph:
    """
    Converts a PyTorch Geometric Data object to a NetworkX graph.

    Args:
        data (Data): The PyTorch Geometric Data object.

    Returns:
        nx.Graph: The corresponding NetworkX graph.
    """
    graph = nx.Graph()

    # Add nodes
    for node, features in enumerate(data.x):
        graph.add_node(node, features=features)

    # Add edges
    for edge in data.edge_index.t().tolist():
        graph.add_edge(edge[0], edge[1])

    return graph