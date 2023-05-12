import numpy as np
import glob
from datetime import datetime
import os
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFilter
import pandas as pd
np.random.seed(0)

def generate_colors(n):
    colors = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
    return colors.tolist()

def combine_masks(mask_folder):
    binary_mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
    combined_mask = None
    for i, binary_mask_file in enumerate(binary_mask_files):
        binary_mask = np.array(Image.open(binary_mask_file))
        if combined_mask is None:
            combined_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        combined_mask[binary_mask == 255] = i + 1
    return combined_mask

def overlay_mask(image, mask, colors, boundary_thickness=3):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for i, color in enumerate(colors):
        colored_mask[mask == (i + 1)] = color
        
    for i in range(1, len(colors)+1 ):
        binary_mask = (mask == i).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_mask, contours, -1, (0, 0, 0), boundary_thickness)
    alpha = 0.8
    colored_mask[mask ==0] = np.array(image)[mask == 0]
    #overlayed = 0.4 * np.array(image) + 0.6 * colored_mask
    overlayed = cv2.addWeighted(np.array(image), 1 - alpha, colored_mask, alpha, 0)
    result = Image.fromarray(overlayed)
    
    return result

def process_image_mask_folder(image_file, mask_folder, colors):
    image = Image.open(image_file)
    mask = combine_masks(mask_folder)
    return overlay_mask(image, mask, colors)

def random_files(folder_path, file_count=10):
    print("Loading all files in the folder...")
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if len(all_files) < file_count:
        print ("There are not enough files in the folder to select the desired number.")
        file_count = len(all_files)
    selected_files = random.sample(all_files, file_count)
    return selected_files

def concat_images_to_plot(image_groups, column_titles, output_path):
    n = len(image_groups)
    k = len(image_groups[0])
    
    if len(column_titles) != k:
        raise ValueError("Number of column titles must match the number of images per group")
    fig, axes = plt.subplots(n, k, figsize=(k*4, n*4))
    colors = color_map()[1:]
    for i, group in enumerate(image_groups):
        for j, img_path in enumerate(group):
            if j != 0 and j!= len(group)-1:
                if "SAM" in img_path:
                    img = vis_sam(group[0], img_path)
                else:    
                    img = vis_cam(group[0], img_path,colors)
            else:    
                img = mpimg.imread(img_path)
            
            axes[i, j].imshow(img, aspect='auto')
            #axes[i, j].set_title(img_path.split("/")[-1][:-4])
            axes[i, j].axis('off')
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title , fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    now = datetime.now().strftime("%Y-%m-%d-%H:%M")
    plt.savefig(os.path.join(output_path, f'{now}.png'),bbox_inches='tight', pad_inches=0)



def vis_sam(image_file, mask_folder):
    num_color = len(os.listdir(mask_folder))+1
    colors = generate_colors(num_color)
    return process_image_mask_folder(image_file, mask_folder, colors)
 


def vis_cam(image_file, mask_file,colors):
    
    image = Image.open(image_file)
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return  overlay_mask(image, mask, colors)

def find_least_10(file_path, column_title, return_column, ascend=True):
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by=column_title, ascending=ascend)
    least_10_rows = df_sorted.head(10)
    return_elements = least_10_rows[return_column].tolist()
    return return_elements



def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

    
 