import json
from build_tree import load_masks, build_hierarchy_tree
import os
def analyze_input_folders(main_folder_path, output_json_file):
    results = {}

    for folder in os.listdir(main_folder_path):
        input_folder_path = os.path.join(main_folder_path, folder)
        if os.path.isdir(input_folder_path):
            masks = load_masks(input_folder_path)
            tree = build_hierarchy_tree(masks)

            branches = {}
            root = 'root'
            for child in tree.successors(root):
                branches[child] = []
                traverse_tree(tree, child, root + " -> " + child, branches[child])
            results[folder] = branches
        print(folder)

    with open(output_json_file, 'w') as outfile:
        json.dump(results, outfile, indent=2)

def traverse_tree(tree, node, branch_str, branches_list):
    children = list(tree.successors(node))
    if not children:
        branches_list.append(branch_str)
    else:
        for child in children:
            traverse_tree(tree, child, branch_str + " -> " + child, branches_list)
            

# Replace 'path/to/your/main_folder' with the actual path containing the input folders
# Replace 'output.json' with the desired name for the output JSON file
analyze_input_folders('/fs/scratch/PAS2099/dataset/SAM-WSSS/SAM/voc12_FT4', 'hierarchy.json')
