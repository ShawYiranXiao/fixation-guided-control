# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
import os
import csv
import random
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

image_folder = os.getenv("DEEPGAZE_IMAGE_DIR", "./data/images")
output_csv = "labels.csv"
temp_output_csv = "labels.csv.tmp"

seed_value = 42
images_per_group = 30

images = [
    f
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._")
]
random.seed(seed_value)
random.shuffle(images)

results = []

# Load previously saved labels
if os.path.exists(output_csv):
    with open(output_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append((row["filename"], row["label"]))


current_index = 0
current_score = None
rest_message_displayed = False

# Tkinter
root = tk.Tk()
root.title("Image Labeling Tool")
root.geometry("1000x960")
root.resizable(False, False)

# ttk styling
style = ttk.Style()
style.theme_use("default")

style.configure(
    "Normal.TButton",
    background="white",
    foreground="black",
    borderwidth=1,
    focusthickness=3,
    focuscolor="none",
)

style.configure("Selected.TButton", background="deepskyblue", foreground="white", borderwidth=2)

style.map(
    "Normal.TButton",
    background=[("active", "white"), ("pressed", "white")],
    foreground=[("active", "black"), ("pressed", "black")],
)

style.map(
    "Selected.TButton",
    background=[("active", "deepskyblue"), ("pressed", "deepskyblue")],
    foreground=[("active", "white"), ("pressed", "white")],
)

image_frame = tk.Frame(root, width=1000, height=800)
image_frame.pack_propagate(False)
image_frame.pack()

img_label = tk.Label(image_frame)
img_label.pack()

buttons = {}

instruction_label = tk.Label(root, text="1 = Noncomplex, 5 = Complex", font=("Helvetica", 14))
instruction_label.pack(pady=10)

score_info_label = tk.Label(root, text="Not labeled yet", font=("Helvetica", 12))
score_info_label.pack(pady=5)

rest_label = tk.Label(root, text="", font=("Helvetica", 14), fg="green")
rest_label.pack(pady=5)

button_frame = tk.Frame(root)
button_frame.pack()

for score in ["1", "2", "3", "4", "5"]:
    button = ttk.Button(button_frame, text=score, width=10, style="Normal.TButton")
    button.pack(side="left", padx=5, pady=5)
    button.bind("<Button-1>", lambda event, s=score: save_label(s))  # Handle left mouse clicks
    buttons[score] = button


def save_results_to_csv():
    """Write results to a temporary file, then replace the destination atomically"""
    tmp_file = temp_output_csv
    with open(tmp_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(results)
    os.replace(tmp_file, output_csv)


def save_label(score):
    global current_score, rest_message_displayed

    img_name = images[current_index]

    #  Replace the existing rating for this image
    results[:] = [(name, sc) for name, sc in results if name != img_name]
    results.append((img_name, score))

    current_score = score
    highlight_selected(score)

    #  Update the displayed rating
    score_info_label.config(text=f"Current label: {score}")

    #  Save after every annotation to preserve progress
    save_results_to_csv()

    #  Clear the break reminder after a rating
    if rest_message_displayed:
        rest_label.config(text="")
        rest_message_displayed = False


def highlight_selected(score):
    for s, button in buttons.items():
        if s == str(score):
            button.config(style="Selected.TButton")
            button.focus_set()
        else:
            button.config(style="Normal.TButton")


def show_image():
    img_path = os.path.join(image_folder, images[current_index])
    img = Image.open(img_path)
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img
    root.title(f"Image {current_index + 1}/{len(images)}")

    highlight_selected(None)  # Clear the highlighted rating

    #  Look up the saved rating
    img_name = images[current_index]
    existing = [sc for name, sc in results if name == img_name]
    if existing:
        score_info_label.config(text=f"Current label: {existing[0]}")
    else:
        score_info_label.config(text="Not labeled yet")

    #  Suggest a break after every 30 images
    global rest_message_displayed
    if current_index > 0 and current_index % images_per_group == 0 and not rest_message_displayed:
        rest_label.config(text="You may take a short break for your eyes.")
        rest_message_displayed = True


def key_handler(event):
    global current_index
    if event.char in ["1", "2", "3", "4", "5"]:
        save_label(event.char)
    elif event.keysym == "Right":
        if current_index < len(images) - 1:
            current_index += 1
            show_image()
    elif event.keysym == "Left":
        if current_index > 0:
            current_index -= 1
            show_image()


root.bind("<Key>", key_handler)

# Launch the interface
show_image()
root.mainloop()
