import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from surgrid.gui_demo.gui_editor_cadisv2 import GUI_Editor_CaDISv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, help='Path to data config yaml file.')
    args = parser.parse_args()
    return args

args = parse_args()

# Main GUI setup
root = tk.Tk()
root.title("CaDISv2 Graph to Image Demo")
root.geometry("3000x1500")

# Styling
style = ttk.Style(root)
style.theme_use('clam')  # or 'alt', 'default', 'classic', 'vista'

main_font = tkfont.Font(family="Helvetica", size=40)
style.configure('TButton', font=main_font, padding=6)
style.configure('TLabel', font=main_font, background='#f0f0f0')

editor = GUI_Editor_CaDISv2(root, config=args.conf, device='cuda')

root.mainloop()
