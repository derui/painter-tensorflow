# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

IMAGE_SIDE = 128
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE * 3
RECORD_SIZE = IMAGE_SIZE * 2


def original_shape():
    return [IMAGE_SIDE, IMAGE_SIDE, 3]


def line_art_shape():
    return [IMAGE_SIDE, IMAGE_SIDE, 3]


class ImagePack(object):
    """
    Provide ability to manage image pair, packing image pair to byte array,
    read original and wire-frame images from packed byte array.
    """

    def __init__(self, pack):
        self.pack_file = pack

    def pack(self, original_file, line_art_file):
        """
        Write binary data as image pair into target.

        An order of pair of images is (original, line_art).
        """

        if not os.path.exists(original_file):
            raise Exception(
                'not found original file: {}'.format(original_file))

        if not os.path.exists(line_art_file):
            raise Exception(
                'not found wire frame file: {}'.format(line_art_file))

        original_image = cv2.imread(original_file, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        line_art_image = cv2.imread(line_art_file, cv2.IMREAD_COLOR)
        line_art_image = cv2.cvtColor(line_art_image, cv2.COLOR_BGR2RGB)

        if original_image is None or line_art_image is None:
            raise Exception('can not read image {},{}'.format(original_file,
                                                              line_art_file))

        ndary = np.concatenate(
            (original_image.reshape([-1]), line_art_image.reshape([-1])))
        self.pack_file.write(np.ndarray.tobytes(ndary))

    def unpack(self, record_index):
        """
        Write binary data as image pair into target.

        An order of pair of images is (original, line_art).
        """
        assert record_index >= 0
        self.pack_file.seek(record_index * RECORD_SIZE, 0)

        array = np.frombuffer(self.pack_file.read(RECORD_SIZE), dtype=np.uint8)
        array = array.reshape([2, IMAGE_SIZE])

        return {'original': array[0], 'line_art': array[1]}
