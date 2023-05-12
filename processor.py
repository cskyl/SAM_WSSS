import os
import numpy as np
from PIL import Image
from merge.max_iou import MaxIoU
from merge.max_iou_imp import MaxIoU_IMP
from merge.max_iou_imp2 import MaxIoU_IMP2
from merge.max_iou_imp import MaxIoU_IMP
from merge.merge_customize import CUSTOMIZE
from pathlib import Path
from collections import OrderedDict
import csv
from concurrent.futures import ThreadPoolExecutor
from util.vis import *

class Processor:
    def __init__(self, params, num_cls=21):
        self.method = params.method
        method_dict = {
            'max_iou': MaxIoU,
            'max_iou_imp': MaxIoU_IMP,
            'max_iou_imp2': MaxIoU_IMP2,
            'merge_customize': CUSTOMIZE,
        }
        if self.method.lower() in method_dict:
            self.merger = method_dict[self.method](params, num_cls=(params.number_class))
            print(f'Using {self.method}')
        else:
            print(f'Invalid method: {self.method}')
        self.params = params
        self.image_list = params.image_list
        self.pseudo_path = params.pseudo_path
        self.sam_path = params.sam_path
        self.gt_dir = params.gt_dir
        tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
        self.save_path = os.path.join(self.pseudo_path.replace('pseudo_label', 'processed_mask'),tag)
        self.eval_path = self.pseudo_path.replace('pseudo_label', 'eval')
        self.vis_path = self.pseudo_path.replace('pseudo_label', 'visualization')
        self.num_cls = num_cls
        self.image_path = params.images_path
        self.vis_sample = params.vis_sample
        self.vis_worse = params.vis_worst
        self.vis_best = params.vis_best
        self.multithread = params.multithread

    def generate_merged_masks(self):
        def process_file(filename, save_path, sam_path, merger):
            name = filename.path.split('.png')[0].split('/')[-1]
            if os.path.isfile(save_path + f"/{name}.png"):
                return
            predict = np.array(Image.open(filename.path))
            sam_folder = os.path.join(sam_path, name)
            merger.merge(predict, name, sam_folder, save_path)
        print("Generating merged masks...")
        if self.multithread:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor() as (executor):
                futures = []
                for filename in os.scandir(self.pseudo_path):
                    if filename.is_file():
                        future = executor.submit(process_file, filename, self.save_path, self.sam_path, self.merger)
                        futures.append(future)
                else:
                    for future in futures:
                        future.result()

        else:
            Path(self.save_path).mkdir(parents=True, exist_ok=True)
            for filename in os.scandir(self.pseudo_path):
                if filename.is_file():
                    name = filename.path.split('.png')[0].split('/')[-1]
                    if os.path.isfile(self.save_path + f"/{name}.png"):
                        pass
                    else:
                        predict = np.array(Image.open(filename.path))
                        sam_folder = os.path.join(self.sam_path, name)
                        self.merger.merge(predict, name, sam_folder, self.save_path)

    def generate_samples(self):
        '''
        given a list of image id, generate a figure with 5 columns
        first: original image
        second: pseudo
        third sam
        fourth processed
        fifth GT
        :param list:
        :return:
        '''
        print("Preparing visualization...")
        if not os.path.exists(self.vis_path):
                os.makedirs(self.vis_path)

        image_rows = []
        selected_files = self.vis_sample
        columns_title = ["'Original'", "'Pseudo'", "'SAM'", "'Processed'", "'GT'"]
        if self.vis_worse:
            print("Visualizing worst samples")
            tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
            csv_path = os.path.join(self.eval_path, f"{tag}_summary.csv")
            selected_files = find_least_10(csv_path, 'mIoU_delta', 'name', True)
        if self.vis_best:
            print("Visualizing best samples")
            tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
            csv_path = os.path.join(self.eval_path, f"{tag}_summary.csv")
            selected_files = find_least_10(csv_path, 'mIoU_delta', 'name', False)
            self.vis_worse = False
            print("Won't process worst samples")
        if selected_files is None or len(selected_files) == 0:
            selected_files = random_files(self.pseudo_path, 10)
            for f in selected_files:
                print('random selected: ', os.path.basename(f))
                image_row = []
                name = f.split('/')[-1].split('.png')[0]
                image_row.append(os.path.join(self.image_path, '%s.jpg' % name))
                image_row.append(f)
                image_row.append(os.path.join(self.sam_path, name))
                image_row.append(os.path.join(self.save_path, '%s.png' % name))
                image_row.append(os.path.join(self.gt_dir, '%s.png' % name))
                image_rows.append(image_row)
        
            tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
            if not os.path.exists(os.path.join(self.vis_path, tag, 'random')):
                os.makedirs(os.path.join(self.vis_path, tag, 'random'))
            concat_images_to_plot(image_rows, columns_title, os.path.join(self.vis_path, tag, 'random'))
        else:
            for f in selected_files:
                image_row = []
                image_row.append(os.path.join(self.image_path, '%s.jpg' % f))
                image_row.append(os.path.join(self.pseudo_path, '%s.png' % f))
                image_row.append(os.path.join(self.sam_path, f))
                image_row.append(os.path.join(self.save_path, '%s.png' % f))
                image_row.append(os.path.join(self.gt_dir, '%s.png' % f))
                image_rows.append(image_row)
                tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
                if self.vis_worse:
                    tag = tag + '/worst'
                elif self.vis_best:
                        tag = tag + '/best'
                if not os.path.exists(os.path.join(self.vis_path, tag)):
                    os.makedirs(os.path.join(self.vis_path, tag))
                concat_images_to_plot(image_rows, columns_title, os.path.join(self.vis_path, tag))
        return

    def evaluate(self):
        Path(self.eval_path).mkdir(parents=True, exist_ok=True)
        tag = '_'.join([os.path.basename(self.pseudo_path), os.path.basename(self.sam_path), os.path.basename(self.method)])
        output_file = os.path.join(self.eval_path, f"{tag}_summary.csv")
        image_index = 0
        if os.path.exists(output_file):
            os.remove(output_file)
        image_index = 0
        for filename in os.scandir(self.save_path):
            if filename.is_file():
                name = filename.path.split('.png')[0].split('/')[-1]
                gt_file = os.path.join(self.gt_dir, '%s.png' % name)
                pseudo_file = os.path.join(self.pseudo_path, '%s.png' % name)
                predict = np.array(Image.open(filename.path))
                gt = np.array(Image.open(gt_file))
                pseudo = np.array(Image.open(pseudo_file))
                cal = gt < 255
                mask_sam = (predict == gt) * cal
                mask_pseudo = (pseudo == gt) * cal
                P_sam, TP_sam, P_pseudo, TP_pseudo, T = 0, 0, 0, 0, 0
                IoU_sam, precision_sam, recall_sam, IoU_pseudo, precision_pseudo, recall_pseudo = [], [], [], [], [], []
                for i in range(1, self.num_cls):
                    true = np.sum((gt == i) * cal)
                    if true == 0:
                        continue

                    # this is share
                    T += true
                    # after sam
                    P_sam += np.sum((predict == i) * cal)
                    TP_sam += np.sum((gt == i) * mask_sam)
                    # before sam
                    P_pseudo += np.sum((pseudo == i) * cal)
                    TP_pseudo += np.sum((gt == i) * mask_pseudo)

                    # after sam
                    IoU_sam.append(TP_sam / (T + P_sam - TP_sam + 1e-10))
                    precision_sam.append(TP_sam / (P_sam + + 1e-10))
                    recall_sam.append(TP_sam / (T + 1e-10))
                    # before sam
                    IoU_pseudo.append(TP_pseudo / (T + P_pseudo - TP_pseudo + 1e-10))
                    precision_pseudo.append(TP_pseudo / (P_pseudo + + 1e-10))
                    recall_pseudo.append(TP_pseudo / (T + 1e-10))

                pseudo_metrics = OrderedDict([('mIoU', round(np.mean(np.array(IoU_pseudo)), 3)),
                                              ('mprecision', round(np.mean(np.array(precision_pseudo)), 3)),
                                              ('mrecall', round(np.mean(np.array(recall_pseudo)), 3))
                                              ])
                sam_metrics = OrderedDict([('mIoU', round(np.mean(np.array(IoU_sam)), 3)),
                                              ('mprecision', round(np.mean(np.array(precision_sam)), 3)),
                                              ('mrecall', round(np.mean(np.array(recall_sam)), 3))
                                              ])

                update_summary(name, pseudo_metrics, sam_metrics, output_file, image_index == 0)

                image_index += 1
        print(f'finish eval {image_index} images')


def update_summary(
        name,
        pseudo_metrics,
        sam_metrics,
        filename,
        write_header=False,
):
    rowd = OrderedDict(name=name)
    rowd.update([('pseudo_' + k, v) for k, v in pseudo_metrics.items()])
    rowd.update([('after_sam_' + k, v) for k, v in sam_metrics.items()])
    rowd.update([('mIoU_delta', sam_metrics['mIoU'] - pseudo_metrics['mIoU'])])
    rowd.update([('mprecision_delta', sam_metrics['mprecision'] - pseudo_metrics['mprecision'])])
    rowd.update([('mrecall_delta', sam_metrics['mrecall'] - pseudo_metrics['mrecall'])])

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)