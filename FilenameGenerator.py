import os
from multiprocessing import Lock as mpLock
import Configuration as cfg



class FilenameGenerator:
    """
    Call FilenameGenerator is used to provide correct filenames for saving results.
    Ensures, that each filename is unique.
    """
    index = 0
    index_lock = None
    image_folder = cfg.get_image_folder()
    data_folder = cfg.get_data_folder()
    sxm_folder = cfg.get_sxm_folder()
    im_pre = cfg.get_prefix_image()
    im_suf = cfg.get_suffix_image()
    da_pre = cfg.get_prefix_data()
    da_suf = cfg.get_suffix_data()
    sx_pre = cfg.get_prefix_sxm()
    sx_suf = cfg.get_suffix_sxm()
    #print(im_pre, im_suf, da_pre, da_suf, sx_pre, sx_suf)

    class AtomicCounter:
        """
        Atomic counter similar to Java.
        Allows for atomic increment and get
        """
        index = 0

        def incAndGet(self):
            """
            Increments index by 1 and returns it
            :return: ++index
            """
            self.index = self.index + 1
            return self.index

        def get(self):
            """
            Only returns the index without incrementing
            :return: index
            """
            return self.index

    def __init__(self, lock=None):
        """
        Initialises new FilenameGenerator
        Lock can be proivided to sync multiple filename generators, but implementation as BaseManager is better
        :param lock:
        """
        self.index = 0
        if lock is None:
            self.index_lock = mpLock()
        else:
            self.index_lock = lock
        self.counter = self.AtomicCounter()

    def generate_Tuple(self):
        """
        Generates a pair of filenames for each filetype
        :return: image_path, data_path, sxm_path, index
        """
        index = self.generateIndex()
        img = str(os.path.join(self.image_folder, self.im_pre + str(index) + self.im_suf))
        dat = str(os.path.join(self.data_folder, self.da_pre + str(index) + self.da_suf))
        sxm = str(os.path.join(self.sxm_folder, self.sx_pre + str(index) + self.sx_suf))
        return img, dat, sxm, index

    def generateIndex(self):
        """
        Generates a new, unique index
        :return: index
        """
        self.index_lock.acquire()
        index = self.counter.incAndGet()
        self.index_lock.release()
        return index

    def peekIndex(self):
        """
        Gets the index without incrementing
        :return: index
        """
        self.index_lock.acquire()
        index = self.counter.get()
        self.index_lock.release()
        return index



