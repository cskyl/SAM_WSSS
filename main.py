import argparse
from processor import Processor

def main():
    args = setup_parser().parse_args()
    processer = Processor(args)
    if args.mode == 'all':
        processer.generate_merged_masks()
        processer.evaluate()
        processer.generate_samples()
    elif args.mode == 'eval':
        processer.evaluate()
    elif args.mode == 'merge':
        processer.generate_merged_masks()
    elif args.mode == 'vis':
        processer.generate_samples()
    else:
        raise NotImplemented

def setup_parser():
    parser = argparse.ArgumentParser(description='SAM_WSSS')

    ########################Pretrained Model#########################
    parser.add_argument('--pseudo_path', type=str, default='pseudo_labels/transcam',
                        help='path to pseudo_label')
    parser.add_argument('--sam_path', type=str, default='SAM/voc12_FT4',
                        help='path to sam')
    parser.add_argument('--image_list',  type=str, default='voc12/train.txt',
                        help='image list')
    parser.add_argument('--method', type=str, default='max_iou_imp2',
                        help='method to merge pseudo label and sam label')
    parser.add_argument('--number_class', default=21,
                        type=int,
                        help='number of class')
    parser.add_argument("--gt_dir", default='VOCdevkit/VOC2012/SegmentationClass', type=str)
    parser.add_argument("--mode", default='merge', choices=['eval', 'merge', 'all', 'vis'], type=str)
    parser.add_argument('--vis_sample', default=None, type=str, nargs='+',
                        help='list of image names to vis')
    parser.add_argument('--images_path', default="VOCdevkit/VOC2012/JPEGImages", type=str,
                        help='original images for visualization')
    parser.add_argument('--vis_worst', action='store_true',
                        help='if true, visualize the worst 10 images')
    parser.add_argument('--vis_best', action='store_true',
                        help='if true, visualize the best 10 images')
    parser.add_argument('--multithread', default=True, type=bool,
                        help='if using multithread in merging')

    return parser


if __name__ == '__main__':
    main()