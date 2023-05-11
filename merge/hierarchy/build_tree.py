import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt

def load_masks(folder_path):
    masks = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            mask = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            masks[filename] = mask
    return masks

def sort_masks_by_area(masks):
    return sorted(masks.items(), key=lambda x: cv2.countNonZero(x[1]), reverse=True)

def overlap_percentage(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    union = cv2.bitwise_or(mask1, mask2)
    return cv2.countNonZero(intersection) / cv2.countNonZero(union)
def coverage_percentage(mask1, mask2):
    intersection = cv2.bitwise_and(mask1, mask2)
    smaller_mask_area = min(cv2.countNonZero(mask1), cv2.countNonZero(mask2))
    return cv2.countNonZero(intersection) / smaller_mask_area
def add_mask_to_tree(filename, mask, tree, node, masks, overlap_threshold):
    for child in tree.successors(node):
        if coverage_percentage(mask, masks[child]) > overlap_threshold:
            return add_mask_to_tree(filename, mask, tree, child, masks, overlap_threshold)
    tree.add_node(filename, parent=node)
    tree.add_edge(node, filename)
    return tree

def build_hierarchy_tree(masks, overlap_threshold=0.5):
    tree = nx.DiGraph()
    root = "root"
    tree.add_node(root)

    sorted_masks = sort_masks_by_area(masks)
    for filename, mask in sorted_masks:
        add_mask_to_tree(filename, mask, tree, root, masks, overlap_threshold)

    return tree

def visualize_levels(tree):
    levels = nx.get_node_attributes(tree, 'parent')
    levels['root'] = 0
    for node in levels:
        levels[node] = 0 if node == "root" else levels[levels[node]] + 1
    max_level = max(levels.values())

    for level in range(max_level + 1):
        print(f"Level {level}:")
        for node in levels:
            if levels[node] == level:
                print(node)
        print()


# Replace 'path/to/your/folder' with the actual folder path containing the binary mask files

masks = load_masks('/fs/scratch/PAS2099/dataset/SAM-WSSS/SAM/voc12_FT4/2008_008525')
tree = build_hierarchy_tree(masks)
# visualize_levels(tree)

def print_branch(tree, node, branch_str):
    children = list(tree.successors(node))
    if not children:
        print(branch_str)
    else:
        for child in children:
            print_branch(tree, child, branch_str + " -> " + child)

def visualize_branches(tree):
    root = 'root'
    print("Branches under root:")
    for child in tree.successors(root):
        print_branch(tree, child, root + " -> " + child)
    print()
    
#visualize_branches(tree)




def get_filenames_at_level(tree, target_level):
    levels = nx.get_node_attributes(tree, 'parent')
    levels['root'] = 0  # Initialize the 'root' key before the loop
    for node in levels:
        if node != 'root':  # Exclude the root node from this loop
            levels[node] = levels[levels[node]] + 1

    filenames_at_level = []
    for node, level in levels.items():
        if level == target_level and node != 'root':
            filenames_at_level.append(node)

    return filenames_at_level
folder = '/fs/scratch/PAS2099/dataset/SAM-WSSS/SAM/voc12_FT4/2008_008525'
print(get_filenames_at_level(tree,0))
import PyQt5 
from ete3 import Tree, TreeStyle

def create_tree_visualization(tree):
    ete_tree = Tree()
    ete_tree_dict = {"root": ete_tree}

    for node, data in tree.nodes(data=True):
        if node != "root":
            parent = data["parent"]
            ete_node = ete_tree_dict[parent].add_child(name=node)
            ete_tree_dict[node] = ete_node

    ts = TreeStyle()
    ts.show_leaf_name = True
    ete_tree.show(tree_style=ts)
    ete_tree.render("mytree.png")
# Replace 'path/to/your/folder' with the actual folder path containing the binary mask files


create_tree_visualization(tree)

