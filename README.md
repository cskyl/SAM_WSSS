# SAM_WSSS
Code repository for our paper "[Segment Anything Model (SAM) Enhanced Pseudo
Labels for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/2305.05803)"
This is a Python script for our proposed framework.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

- Python 3.6 or higher

### Installation

Clone the repository:

```bash
git https://github.com/cskyl/SAM_WSSS.git
cd SAM_WSSS
```

### Running mask enhancement

You can specify the locations of your pseudo labels and SAM masks folders by modifying the pseudo_path and sam_path respectively. Use the command below, replacing your_pseudo_labels and your_SAM_masks with the corresponding paths on your system:

```
python main.py --pseudo_path <your_pseudo_labels> --sam_path <your_SAM_masks> 
```

This will use our default merging algorithm, and the result will be stored in 
```
SAM_WSSS/processed_masks
```

### Evaluation and Visualization

To evaluate the quality of the pseudo masks and enhanced masks, you can use the following command. Please note that the current version of our code supports evaluation for object classes from the PASCAL VOC 2012 dataset.
```
python main.py --mode eval --pseudo_path <your_pseudo_labels> --sam_path <your_SAM_masks> --gt_dir <VOCdevkit/VOC2012/JPEGImages>
```

This script will generate evaluation statistics for each image sample, which will be saved in a .csv file in the following directory:
```
SAM_WSSS/eval
```

For visualization purposes, you can use the command below. You have the option to visualize specific samples by using vis_sample. Alternatively, you can use vis_best or vis_worst to visualize the top 10 samples that showed the most improvement or the least improvement, respectively, in delta mIoU after enhancement.
```
python main.py --mode vis --pseudo_path <your_pseudo_labels> --sam_path <your_SAM_masks> --gt_dir <VOCdevkit/VOC2012/JPEGImages> --images_path <VOCdevkit/VOC2012/JPEGImages>
```
This will generat the visualization in
```
SAM_WSSS/visualizations
```

## Customizing the Merging Algorithm

If you wish to implement your own merging algorithm, you can modify the `merge` function located in the following script:
```
SAM_WSSS/merge/merge_customize
```
To run your custom algorithm, use the following command:
```
python main.py --pseudo_path <your_pseudo_labels> --sam_path <your_SAM_masks> --method 'merge_customize'
```

By changing the --mode argument to 'all', you can run the entire pipeline, which includes merging, evaluation, and visualization. 


If you are using our code, please consider citing our paper.

```
```
