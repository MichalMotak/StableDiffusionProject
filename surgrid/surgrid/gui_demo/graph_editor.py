import torch
import tkinter as tk

from surgrid.gui_demo.utils import rgb_to_hex
from surgrid.gui_demo.graph import Graph


class Graph_Editor:

    def __init__(self, root: tk.Tk, canvas, gt_image_label, legend_frame):
        self.root = root
        self.canvas = canvas
        self.gt_image_label = gt_image_label
        self.legend_frame = legend_frame
        self.graph = Graph()

        self.node_radius = 10
        self.drag_data = {"x": 0, "y": 0, "item": None}

        self.resize_job = None  # To keep track of the scheduled resize job

        self.next_node_id = 0
        self.node_id_to_canvas_item = {}
        self.canvas_item_to_node_id = {}

        self.classes = {
            'Undefined': (0., 0., 0.),  # Black
        }
        self.n_classes = 1

    def update_legend(self, classes):
        # Clear the current legend
        for widget in self.legend_frame.winfo_children():
            widget.destroy()

        # Add a title to the legend
        title = tk.Label(self.legend_frame, text="Node Classes", bg='#f0f0f0')
        title.pack(pady=(0, 10))

        # Populate the legend with class names and color indicators
        for class_name, color in classes.items():
            color = rgb_to_hex(color)
            class_frame = tk.Frame(self.legend_frame, bg='#f0f0f0')
            class_frame.pack(fill='x', padx=5, pady=2)

            # Create a small canvas for each color indicator
            node_indicator = tk.Canvas(class_frame, width=20, height=20, bg='#f0f0f0', bd=0, highlightthickness=0)
            node_indicator.pack(side='left', padx=(0, 10))

            # Draw a circle on the canvas to represent a node
            radius = 8  # Adjust the size of the node representation as needed
            node_indicator.create_oval(10 - radius, 10 - radius, 10 + radius, 10 + radius, fill=color)  # , outline=color)

            class_label = tk.Label(class_frame, text=class_name, bg='#f0f0f0')
            class_label.pack(side='left')

    def create_node(self, features: torch.Tensor, color: tuple) -> None:
        """
        Create a new node at the specified relative location.

        :param features: The features of the node.
        :param color: The color of the node.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def change_node_class(self, item: int, new_class_ft: torch.Tensor, color: str) -> None:
        """
        Change the class (and color) of the selected node.

        :param item: The ID of the node to change.
        :param color: The new color of the node.
        """
        self.canvas.itemconfig(item, fill=color)
        self.graph.change_node_class(self.canvas_item_to_node_id[item], new_class_ft)

    def delete_node(self, item: int) -> None:
        """
        Delete a node.

        :param item: The ID of the node to delete.
        """
        self.canvas.delete(item)
        self.graph.remove_node(self.canvas_item_to_node_id[item])

        # Remove edges associated with the node
        self.graph.edges = [edge for edge in self.graph.edges if self.canvas_item_to_node_id[item] not in edge]

        # Redraw the graph to reflect the changes
        self.draw_edges()

        del self.node_id_to_canvas_item[self.canvas_item_to_node_id[item]]
        del self.canvas_item_to_node_id[item]
        # self.next_node_id -= 1

    def on_node_move(self, event: tk.Event) -> None:
        """
        Handle the node move event by updating the node's position based on mouse movement.
        Used for drag-and-drop of nodes.

        :param event: The event information containing the mouse position.
        """
        if not self.drag_data["item"]:
            return  # Exit if no node is selected

        # Calculate new relative positions based on mouse movement
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate the relative displacement
        rel_dx = dx / canvas_width
        rel_dy = dy / canvas_height

        self.graph.move_node(self.canvas_item_to_node_id[self.drag_data["item"]], rel_dx, rel_dy)
        self.canvas.move(self.drag_data["item"], dx, dy)
        self.draw_edges()

        # Update the drag data with the current mouse position for the next movement calculation
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_node_press(self, event: tk.Event) -> None:
        """
        Handle the node press event.

        :param event: The event information.
        """

        # Record the item and its location
        closest_items = self.canvas.find_closest(event.x, event.y)
        if closest_items and "node" in self.canvas.gettags(closest_items[0]):
            self.drag_data["item"] = closest_items[0]
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_node_release(self, event: tk.Event) -> None:
        """
        Handle the node release event.

        :param event: The event information.
        """
        # Reset the drag information
        self.drag_data = {"x": 0, "y": 0, "item": None}

    def update_all_node_positions(self) -> None:
        """
        Update the positions of all nodes based on their stored relative positions.
        Used when canvas is resized.
        Keeps the nodes in the same relative positions.
        """
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        for node_id, (rel_x, rel_y) in self.graph.get_all_node_coords():
            node_id = self.node_id_to_canvas_item[node_id]
            # Convert the relative positions directly to new absolute positions
            new_x = rel_x * canvas_width
            new_y = rel_y * canvas_height

            # Update the canvas item's position
            self.canvas.coords(node_id, new_x - self.node_radius, new_y - self.node_radius,
                               new_x + self.node_radius, new_y + self.node_radius)

    def on_canvas_resize(self, event: tk.Event) -> None:
        """
        Handle the canvas resize event.

        :param event: The event information.
        """

        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(1, self.update_all_node_positions)
        self.draw_edges()

    def draw_edges(self):
        """
        Draw edges on the canvas for each pair of connected nodes.

        :param edges: A list of tuples, where each tuple represents an edge between two nodes.
        """
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        # First, delete all existing lines (edges) from the canvas
        self.canvas.delete("edge")  # Assuming 'edge' tag is used for all edge lines

        for edge in self.graph.edges:
            node_id1, node_id2 = edge
            # Retrieve the center positions of the nodes
            x1, y1 = self.graph.get_node_coords(node_id1)  # TKinter node IDs start at 1
            x2, y2 = self.graph.get_node_coords(node_id2)

            # Draw a line between the nodes
            edge_id = self.canvas.create_line(x1 * canvas_width,
                                              y1 * canvas_height,
                                              x2 * canvas_width,
                                              y2 * canvas_height,
                                              fill='black',
                                              tags="edge")
            self.canvas.lower(edge_id)  # Lower the edge line to be behind the nodes

    def load_graph(self) -> None:
        """
        Load a graph from a file.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
