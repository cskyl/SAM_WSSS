from merge.merge_base import Merger
import os
from PIL import Image
import numpy as np
from pathlib import Path

class MaxIoU_IMP(Merger):
    def __init__(self, params, num_cls=21, threshold=0.2):
        super(MaxIoU_IMP, self).__init__(params, num_cls)
        self.threshold = threshold

    def merge(self, predict, name, sam_folder, save_path):
        seen = []
        processed_mask = np.zeros_like(predict)
       
        for i in range(1, self.num_cls):
            pre_cls = predict == i
            if np.sum(pre_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pre_cls)
            for filename in os.scandir(sam_folder):
                if filename.is_file() and filename.path.endswith('png') and filename.path not in seen:
                    cur = np.array(Image.open(filename.path)) == 255
                    sam_mask = np.logical_or(sam_mask, cur)
                    improve = 2 * np.sum((pre_cls == cur) * pre_cls) - np.sum(cur)
                    # overlap_ratio = np.sum((pre_cls == cur) * pre_cls) / np.sum(pre_cls)
                    # if improve > 0 or overlap_ratio >= self.threshold:
                    
                    if improve > 0:
                        
                        candidates.append(cur)
                        seen.append(filename.path)
                        iou += np.sum(pre_cls == cur)
            sam_mask = np.logical_and(sam_mask==0, pre_cls==1)
            candidates.append(sam_mask)
            processed_mask[np.sum(candidates, axis=0) > 0] = i
        
        im = Image.fromarray(processed_mask)
        im.save(f'{save_path}/{name}.png')