import abc
from abc import abstractmethod


class Merger(metaclass=abc.ABCMeta):
    def __init__(self, params, num_cls=21):
        self.params = params
        self.num_cls = num_cls


    @abstractmethod
    def merge(self, predict, name, sam_folder, save_path):
        pass



