import torch
from torch_geometric.data import Data


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

        self.edge_index_to_node_id = {}
        self.node_id_to_edge_index = {}

    """
    def add_node(self, node_id, relative_x, relative_y):
        self.nodes[node_id] = (relative_x, relative_y)
        print(f"Added node {node_id} at ({relative_x}, {relative_y})")
    """

    def generate_continuous_index_mapping(self):
        """Generate a mapping from current node IDs to a continuous range of indices."""
        sorted_node_ids = sorted(self.nodes.keys())
        return {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}

    def add_node(self, node_id: int, features: torch.Tensor) -> None:
        self.nodes[node_id] = features
        print(f"Added node {node_id} at ({features[-2].item()}, {features[-1].item()})")

    def remove_node(self, node_id: int) -> None:
        # Remove the node from the nodes dictionary
        if node_id in self.nodes:
            del self.nodes[node_id]
            print(f"Removed node {node_id}")
            self.edges = [(src, dest) for src, dest in self.edges if src != node_id and dest != node_id]
        else:
            print(f"Node {node_id} not found")

    def move_node(self, node_id: int, rel_dx: float, rel_dy: float) -> None:
        self.nodes[node_id][-2] += rel_dx
        self.nodes[node_id][-1] += rel_dy


        print(f"Moved node {node_id} to ({self.nodes[node_id][-2].item()}, {self.nodes[node_id][-1].item()})")

    def change_node_class(self, node_id: int, new_class_ft: torch.Tensor) -> None:
        self.nodes[node_id][:-4] = new_class_ft
        print(f"Changed class of node {node_id} to {new_class_ft.tolist()}")

    def get_node(self, node_id: int) -> torch.Tensor:
        return self.nodes[node_id]

    def get_node_coords(self, node_id: int) -> (float, float):
        coords = self.nodes[node_id][-2:]
        return coords[0].item(), coords[1].item()

    def get_all_nodes(self) -> dict:
        return self.nodes

    def get_all_node_coords(self):
        coord_list = [self.get_node_coords(node_id) for node_id in self.nodes.keys()]
        return zip(self.nodes.keys(), coord_list)

    def to_torch_geometric(self) -> Data:
        """
        Convert the graph to a torch_geometric.Data format.

        :return: torch_geometric.Data object representing the graph.
        """
        index_mapping = self.generate_continuous_index_mapping()

        # Convert node features to a tensor, maintaining the order of indices
        node_features = [self.nodes[node_id].tolist() for node_id in sorted(self.nodes.keys())]
        x = torch.tensor(node_features, dtype=torch.float)

        # Re-index edges based on the continuous index mapping
        edge_index = [[index_mapping[src], index_mapping[dest]] for src, dest in self.edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


        return Data(x=x, edge_index=edge_index)

