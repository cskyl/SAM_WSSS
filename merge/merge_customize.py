from merge.merge_base import Merger
import os
from PIL import Image
import numpy as np
from pathlib import Path

class CUSTOMIZE(Merger):
    def __init__(self, params, num_cls=21, threshold=0.2):
        super(CUSTOMIZE, self).__init__(params, num_cls)
        self.threshold = threshold

    def merge(self, predict, name, sam_folder, save_path):
        seen = []
        processed_mask = np.zeros_like(predict)
       
        ## customize your own merge method here
        
        im = Image.fromarray(processed_mask)
        im.save(f'{save_path}/{name}.png')