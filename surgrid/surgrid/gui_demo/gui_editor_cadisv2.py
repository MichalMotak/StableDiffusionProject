import os
import torch
import torch.nn.functional as F
import tkinter as tk
import numpy as np
from torchvision.utils import make_grid
from PIL import Image, ImageTk
from datetime import datetime

from surgrid.dataset.cadis_utils import get_cadis_float_cmap, visualize_graph_list
from surgrid.gui_demo.gui_editor import GUI_Editor
from surgrid.diffusion.sampler import Sampler

display_shape = (3, 270, 480)

class GUI_Editor_CaDISv2(GUI_Editor):
    def __init__(self,
                 root: tk.Tk,
                 config: str,
                 device: str = 'cuda'):

        super().__init__(root=root, dataset='CaDISv2', device=device)
        
        self.sg2img = Sampler(config)
        for id, exp in enumerate(["EXP1", "EXP2", "EXP3"]):
            btn = tk.Button(self.root, text=exp,
                            command=lambda e=exp: self.graph_editor.set_active_experiment(e))
            btn.grid(row=3 + id, column=0, padx=5, pady=10, sticky='ew')

        btn = tk.Button(self.root, text="save graph",
                        command=lambda e=exp: self.save_graph())
        btn.grid(row=3 + 3, column=0, padx=5, pady=10, sticky='ew')

    def save_graph(self):
        print("elo")


        graph_data = self.graph_editor.graph.to_torch_geometric()
        print(graph_data.x)

        graph_data.x = graph_data.x.to(torch.float32)
        # graph_data.x[:, -4:] = (graph_data.x[:, -4:] + 1.0) / 2.0  # Normalise the node features to [0, 1]

        print(f"{graph_data=}")
        print(f"{graph_data.x=}")

        
        save_path = f'results/gui_samples/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}/'
        os.makedirs(save_path, exist_ok=False)
        # Save the generated image to hard drive
        visualize_graph_list(data_list=[graph_data],
                             image_shape=(150, 180),
                             num_classes=self.graph_editor.n_classes,
                             class_names=list(self.graph_editor.classes.keys()),
                             palette=get_cadis_float_cmap(),
                             save_path=os.path.join(save_path, 'graph.svg')
                             )
        
        torch.save(graph_data, os.path.join(save_path, 'graph.pt'))


    def generate_image_from_graph(self, nodes) -> Image:
        """
        Generate an image from a graph.

        :param nodes: The nodes of the graph.
        :return: The generated image.
        """
        # Your model code will go here
        # For demonstration, just return a blank image
        return Image.new('RGB',
                         (display_shape[-1], display_shape[-2]),
                         color='lightgrey')

    def on_generate_button_pressed(self) -> None:
        """
        Handle the generate button press event.

        TODO: Separate model loading from data generation, e.g. with additional button.

        TODO: Save the generated images and current graph to hard drive in .svg format
        """

        assert self.graph_editor.n_classes is not None

        # Convert current tkinter graph to torch_geometric graph data
        graph_data = self.graph_editor.graph.to_torch_geometric()
        print(graph_data.x)

        graph_data.x = graph_data.x.to(torch.float32)
        # graph_data.x[:, -4:] = (graph_data.x[:, -4:] + 1.0) / 2.0  # Normalise the node features to [0, 1]

        print(f"{graph_data=}")
        # Normalise continuous node features
        generated_image = self.sg2img.scenegraph_to_image(graph_data, cond_scale = 1.5, batch_size = 4)

        print(f"{generated_image.shape=}")
        # Un-normalise node features
        # graph_data.x[:, -4:] = (graph_data.x[:, -4:] * 2.0) - 1.0

        generated_image = F.interpolate(generated_image,
                                        size=(np.array(display_shape[1:])//2).tolist(),
                                        mode='bilinear')
        grid = make_grid(generated_image, nrow=np.floor(np.sqrt(generated_image.shape[0])).astype(int))
        print(f"{grid.shape=}")
        # generated_image = (generated_image * 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        # print(f"{generated_image.shape=}")
        # img = Image.fromarray(generated_image)
        grid = (grid * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(grid)
        photo_img = ImageTk.PhotoImage(img)
        # img = ImageTk.PhotoImage(generated_image[0].permute(1, 2, 0).cpu().numpy())
        self.gen_image_label.config(image=photo_img)
        self.gen_image_label.image = photo_img

        save_path = f'results/gui_samples/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}/'
        os.makedirs(save_path, exist_ok=False)


        # Save the generated image to hard drive
        visualize_graph_list(data_list=[graph_data],
                             image_shape=(150, 180),
                             num_classes=self.graph_editor.n_classes,
                             class_names=list(self.graph_editor.classes.keys()),
                             palette=get_cadis_float_cmap(),
                             save_path=os.path.join(save_path, 'graph.svg')
                             )
        
        img.save(os.path.join(save_path, 'original_image.png'))
        for n in range(generated_image.shape[0]):
            img_n = (generated_image[n] * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            img_n = Image.fromarray(img_n)
            img_n.save(os.path.join(save_path, f'generated_image_{n}.png'))
