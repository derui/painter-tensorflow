# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import logging
import random

IMAGE_SIDE = 128
IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE * 3
LINE_ART_SIZE = IMAGE_SIDE * IMAGE_SIDE
RECORD_SIZE = IMAGE_SIZE + LINE_ART_SIZE


def original_shape():
    return [IMAGE_SIDE, IMAGE_SIDE, 3]


def line_art_shape():
    return [IMAGE_SIDE, IMAGE_SIDE, 1]


class ImagePack(object):
    """
    Provide ability to manage image pair, packing image pair to byte array,
    read original and wire-frame images from packed byte array.
    """

    def __init__(self, pack):
        self.pack_file = pack

    def pack(self, original_file, wire_frame_file):
        """
        Write binary data as image pair into target.

        An order of pair of images is (original, wire_frame).
        """

        if not os.path.exists(original_file):
            raise Exception(
                'not found original file: {}'.format(original_file))

        if not os.path.exists(wire_frame_file):
            raise Exception(
                'not found wire frame file: {}'.format(wire_frame_file))

        original_image = cv2.imread(original_file, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        wire_frame_image = cv2.imread(wire_frame_file, cv2.IMREAD_GRAYSCALE)

        if original_image is None or wire_frame_image is None:
            raise Exception('can not read image {},{}'.format(original_file,
                                                              wire_frame_file))

        ndary = np.concatenate(
            (original_image.reshape([-1]), wire_frame_image.reshape([-1])))
        self.pack_file.write(np.ndarray.tobytes(ndary))

    def unpack(self, record_index):
        """
        Write binary data as image pair into target.

        An order of pair of images is (original, wire_frame).
        """
        assert record_index >= 0
        self.pack_file.seek(record_index * RECORD_SIZE, 0)

        array = np.frombuffer(self.pack_file.read(RECORD_SIZE), dtype=np.uint8)
        array = array.reshape([2, IMAGE_SIZE])

        return array
