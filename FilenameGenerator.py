import os
from multiprocessing import Lock as mpLock
import Configuration as cfg



class FilenameGenerator:
    index = 0
    index_lock = None
    image_folder = cfg.get_image_folder()
    data_folder = cfg.get_data_folder()
    im_pre = cfg.get_prefix_image()
    im_suf = cfg.get_suffix_image()
    da_pre = cfg.get_prefix_data()
    da_suf = cfg.get_suffix_data()

    class AtomicCounter:
        index = 0

        def incAndGet(self):
            self.index = self.index + 1
            return self.index

    def __init__(self, lock=None):
        self.index = 0
        if lock is None:
            self.index_lock = mpLock()
        else:
            self.index_lock = lock
        self.counter = self.AtomicCounter()

    def generate_Tuple(self):
        index = self.generateIndex()
        img = str(os.path.join(self.image_folder, self.im_pre + str(index) + self.im_suf))
        dat = str(os.path.join(self.data_folder, self.da_pre + str(index) + self.da_suf))
        return img, dat, index

    def generateIndex(self):
        self.index_lock.acquire()
        index = self.counter.incAndGet()
        self.index_lock.release()
        return index
