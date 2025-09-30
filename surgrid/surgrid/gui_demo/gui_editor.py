import torch
import tkinter as tk
from tkinter import Menu
from PIL import Image

from surgrid.gui_demo.graph_editor_cadisv2 import Graph_Editor_CaDISv2
from surgrid.gui_demo.utils import rgb_to_hex

display_shape = (3, 270, 480)

class GUI_Editor:
    def __init__(self,
                 root: tk.Tk,
                 dataset: str,
                 device: str = 'cuda'):
        self.root = root

        # Layout configuration
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)

        # self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas = tk.Canvas(root,
                                width=display_shape[1],
                                height=display_shape[2],
                                bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_node_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_node_release)
        self.canvas.bind("<B1-Motion>", self.on_node_move)
        self.canvas.bind("<Button-3>", self.on_right_click)
        """
        for id, exp in enumerate(["EXP1", "EXP2", "EXP3"]):
            btn = tk.Button(self.root, text=exp,
                            command=lambda e=exp: self.graph_editor.set_active_experiment(e))
            btn.grid(row=3 + id, column=0, padx=5, pady=10, sticky='ew')
        """
        # Image display
        # Frame for the generated image
        self.gen_image_frame = tk.Frame(self.root, bg='#f0f0f0', bd=2, relief='groove')
        self.gen_image_frame.grid(row=0, column=2, padx=10, pady=5, sticky="nsew", rowspan=1)
        self.gen_image_label = tk.Label(self.gen_image_frame, text="Generated Images Here", background='white')
        self.gen_image_label.pack(fill=tk.BOTH, expand=True)
        # Frame for the ground truth image
        self.gt_image_frame = tk.Frame(self.root, bg='#f0f0f0', bd=2, relief='groove')
        self.gt_image_frame.grid(row=1, column=2, padx=10, pady=5, sticky="nsew", rowspan=5)
        self.gt_image_label = tk.Label(self.gt_image_frame, text="Ground Truth Image Here", background='white')
        self.gt_image_label.pack(fill=tk.BOTH, expand=True)

        # Frame for color/node legend
        self.legend_frame = tk.Frame(root, bg='#f0f0f0', bd=2, relief='groove')
        self.legend_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew", rowspan=6)

        self.current_menu = None  # Track the current open context menu

        self.dataset = dataset
        if dataset == 'CaDISv2':
            self.graph_editor = Graph_Editor_CaDISv2(root, self.canvas, self.gt_image_label, self.legend_frame)
        elif dataset == 'CholecSeg8k':
            self.graph_editor = Graph_Editor_CholecSeg8k(root, self.canvas, self.gt_image_label, self.legend_frame)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented.")

        # Generate Image button
        self.generate_button = tk.Button(root, text="Generate Image", command=self.on_generate_button_pressed)
        self.generate_button.grid(row=1, column=0, padx=5, pady=10, sticky='ew')

        # Load Graph button
        self.load_button = tk.Button(root, text="Load Graph", command=self.graph_editor.load_graph)
        self.load_button.grid(row=2, column=0, padx=5, pady=10, sticky='ew')

        # Load models
        self.vq_gan, self.ldm, self.sampler, self.gvae, self.graph_emb = None, None, None, None, None

        self.device = device

    def generate_image_from_graph(self, nodes) -> Image:
        """
        Generate an image from a graph.

        :param nodes: The nodes of the graph.
        :return: The generated image.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def on_generate_button_pressed(self) -> None:
        """
        Handle the generate button press event.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def on_right_click(self, event: tk.Event) -> None:
        """
        Handle the right click event.

        :param event: The event information.
        """
        if self.current_menu:
            self.current_menu.unpost()
        self.current_menu = Menu(self.root, tearoff=0)

        # Flag to check if the right-click was on any node
        click_on_node = False

        # Iterate over all nodes to find if the click is on any of them
        for node_id, (rel_x, rel_y) in self.graph_editor.graph.get_all_node_coords():
            node_id = self.graph_editor.node_id_to_canvas_item[node_id]
            # Convert relative positions to canvas coordinates
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            center_x = rel_x * canvas_width
            center_y = rel_y * canvas_height

            # Calculate the bounding box of the node
            x1, y1 = center_x - self.graph_editor.node_radius, center_y - self.graph_editor.node_radius
            x2, y2 = center_x + self.graph_editor.node_radius, center_y + self.graph_editor.node_radius

            # Check if the click is within the bounding box of the node
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                click_on_node = True
                # Context menu for the node
                # Add 'Change Class' menu with a submenu for class choices
                change_class_menu = Menu(self.current_menu, tearoff=0)
                self.current_menu.add_cascade(label="Change Class", menu=change_class_menu)
                self.current_menu.add_command(label="Delete Node",
                                              command=lambda: self.graph_editor.delete_node(node_id))
                # Populate the submenu with class options
                for class_id, (class_name, color) in enumerate(self.graph_editor.classes.items()):
                    if self.dataset == 'CaDISv2':
                        class_id_ft = torch.zeros(self.graph_editor.n_classes - 1)  # w/o ignore class
                    else:
                        class_id_ft = torch.zeros(self.graph_editor.n_classes)
                    class_id_ft[class_id] = 1
                    change_class_menu.add_command(
                        label=class_name,
                        command=lambda item=node_id, ft=class_id_ft, clr=rgb_to_hex(color):
                            self.graph_editor.change_node_class(item, ft, clr)
                    )
                break  # Node found, no need to check further

        if not click_on_node:
            # Right-click not on a node, option to create a node at the clicked position
            rel_x = event.x / self.canvas.winfo_width()
            rel_y = event.y / self.canvas.winfo_height()
            if self.dataset == 'CaDISv2':
                ft = torch.tensor([0.]*(self.graph_editor.n_classes - 1) + [0.1, 0.1, rel_x, rel_y])
            else:
                ft = torch.tensor([0.]*self.graph_editor.n_classes + [0.1, 0.1, rel_x, rel_y])
            self.current_menu.add_command(label="Create Node",
                                          command=lambda: self.graph_editor.create_node(ft, (0., 0., 0.)))

        self.current_menu.post(event.x_root, event.y_root)

    def on_node_press(self, event: tk.Event) -> None:
        """
        Handle the node press event.

        :param event: The event information.
        """
        # Close any open context menu
        if self.current_menu:
            self.current_menu.unpost()
            self.current_menu = None
        self.graph_editor.on_node_press(event)

    def on_node_release(self, event: tk.Event) -> None:
        """
        Handle the node release event.

        :param event: The event information.
        """
        self.graph_editor.on_node_release(event)

    def on_node_move(self, event: tk.Event) -> None:
        """
        Handle the node move event.

        :param event: The event information.
        """
        self.graph_editor.on_node_move(event)

    def on_canvas_resize(self, event: tk.Event) -> None:
        """
        Handle the canvas resize event.

        :param event: The event information.
        """
        self.graph_editor.on_canvas_resize(event)
