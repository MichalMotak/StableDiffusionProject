import PIL.Image
import torch
import tkinter as tk
from tkinter import filedialog
from torch_geometric.data import Data
from PIL import ImageTk

from surgrid.gui_demo.graph_editor import Graph_Editor
from surgrid.gui_demo.utils import data_to_colors_cadisv2, rgb_to_hex, EXP1, EXP2, EXP3, get_cadis_float_cmap, gt_filepath_to_img_path_cadisv2

display_shape = (3, 270, 480)
class Graph_Editor_CaDISv2(Graph_Editor):

    def __init__(self, root: tk.Tk, canvas, gt_image_label, legend_frame):

        super().__init__(root, canvas, gt_image_label, legend_frame)

        self.active_exp = tk.StringVar(value="")  # No experiment is active initially

    def set_active_experiment(self, exp: str):
        self.active_exp.set(exp)
        self.update_classes_based_on_exp(exp)
        self.update_button_states()

    def update_button_states(self):
        for widget in self.root.grid_slaves(row=3):
            if isinstance(widget, tk.Button):
                widget.config(relief="sunken" if self.active_exp.get() == widget.cget("text") else "raised")

    def update_classes_based_on_exp(self, exp: str):
        self.classes = {}
        cmap = get_cadis_float_cmap()
        if exp == "EXP1":
            for i in EXP1["CLASS"].keys():
                self.classes[EXP1["CLASS"][i]] = cmap[i].tolist()
                self.n_classes = 9
        elif exp == "EXP2":
            for i in list(EXP2["CLASS"].keys())[:-1]:
                self.classes[EXP2["CLASS"][i]] = cmap[i].tolist()
                self.n_classes = 18
        elif exp == "EXP3":
            for i in list(EXP3["CLASS"].keys())[:-1]:
                self.classes[EXP3["CLASS"][i]] = cmap[i].tolist()
                self.n_classes = 26
        else:
            self.classes = {}
        print(f"Active Experiment: {exp}, Classes: {self.classes}")

    def create_node(self, features: torch.Tensor, color: tuple) -> None:
        """
        Create a new node at the specified relative location.

        :param features: The features of the node.
        :param color: The color of the node.
        """
        if self.active_exp.get() == "":
            print("No active experiment selected.")
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        node_id = self.next_node_id
        self.next_node_id += 1
        relative_x = features[-2].item()
        relative_y = features[-1].item()
        canvas_item_id = self.canvas.create_oval(relative_x * canvas_width - self.node_radius,
                                                 relative_y * canvas_height - self.node_radius,
                                                 relative_x * canvas_width + self.node_radius,
                                                 relative_y * canvas_height + self.node_radius,
                                                 fill=rgb_to_hex(color), tags="node")

        self.graph.add_node(node_id, features)
        self.canvas_item_to_node_id[canvas_item_id] = node_id
        self.node_id_to_canvas_item[node_id] = canvas_item_id

    def load_graph(self) -> None:
        """
        Load a graph from a file.
        """
        # Open file dialog to select a graph file
        filepath = filedialog.askopenfilename()
        if not filepath:
            return

        # Load the graph (this is a placeholder, replace with your actual graph loading logic)
        graph = torch.load(filepath)  # Assuming the graph is saved using torch.save

        # Check if the graph is in the correct format
        if not isinstance(graph, Data):
            print("The loaded graph is not in torch_geometric Data format.")
            return

        # Clear existing nodes and edges
        self.canvas.delete("node")
        self.canvas.delete("edge")
        self.graph.nodes.clear()
        self.graph.edges = []
        self.canvas.update()
        self.next_node_id = 0

        # Obtain class id from filename
        if "exp1" in filepath.lower():
            self.set_active_experiment("EXP1")
            exp_id = 1
        elif "exp2" in filepath.lower():
            self.set_active_experiment("EXP2")
            exp_id = 2
        elif "exp3" in filepath.lower():
            self.set_active_experiment("EXP3")
            exp_id = 3
        else:
            raise ValueError("Could not determine experiment id from filename.")
        node_colors = data_to_colors_cadisv2(graph, exp_id=exp_id)

        # Extract and display nodes from the graph
        for n in range(graph.num_nodes):
            assert 0 <= graph.x[n][-2] <= 1 and 0 <= graph.x[n][-1] <= 1, \
                "Node coordinates must be normalized to [0, 1]"
            self.create_node(graph.x[n], node_colors[n].tolist())

        # Draw edges
        self.graph.edges = graph.edge_index.t().tolist()
        # self.draw_edges(graph.edge_index.t().tolist())
        self.draw_edges()

        # Display the ground truth image
        img = PIL.Image.open(gt_filepath_to_img_path_cadisv2(filepath))
        img = img.resize((128,128),
                    PIL.Image.BILINEAR)
        img = img.resize((display_shape[-1]//2,
                          display_shape[-2]//2),
                         PIL.Image.BILINEAR)
        img = ImageTk.PhotoImage(img)
        self.gt_image_label.config(image=img)
        self.gt_image_label.image = img  # Keep a reference to avoid garbage collection

        # Update the legend frame
        self.update_legend(self.classes)
