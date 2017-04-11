# -*- coding: utf-8 -*-

from . import dataset as ip
import numpy as np
import os
import random


class DataSetReader(object):
    """
    DataSetReader reads images from dataset files in directory given as arguments.
    
    Specification of dataset read via this is follows.

    - Each data length is 512*512*3*2 byte as pair of original image and wire-framed image.
    - First of each data pair is original image.
    - Second of each data pair is wire-framed image.
    - all images are flattened with numpy's reshape method.
      - applied np.reshape(512*512*3), so aligns of data is RGBRGBRGB...
    - A data pack contains 1000 image pairs.

    """

    def __init__(self, dataset_dir):
        assert os.path.exists(dataset_dir)

        self.dataset_dir = dataset_dir

        self.__pack_name_format = 'image_pack_{}.bin'

        self.__datasets = []
        for root, _, files in os.walk(dataset_dir):
            self.__datasets.extend([(os.path.join(root, f), os.stat(os.path.join(root,f))) for f in files])

    def inputs(self, batch_size):

        images = np.zeros([2, batch_size, ip.IMAGE_SIZE], np.float32)
        for i in range(batch_size):
            findex = random.randrange(len(self.__datasets))
            f, stat = self.__datasets[findex]
            rindex = random.randrange(stat.st_size / ip.RECORD_SIZE)

            with open(f, "rb") as fh:
                pack = ip.ImagePack(fh)
                packed = pack.unpack(rindex)
                images[0, i] = self._preprocess(packed['original'])
                images[1, i] = self._preprocess(packed['line_art'])

        return np.reshape(images, [2, batch_size, ip.IMAGE_SIDE, ip.IMAGE_SIDE, 3])

    def _preprocess(self, image):
        image = np.ndarray.astype(image, np.float32)
        image = np.multiply(image, 1 / 255.0)
        image = np.multiply(image, 2) - 1.0

        return image
