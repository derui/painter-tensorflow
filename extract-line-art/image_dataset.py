# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import logging
import random

IMAGE_SIZE = 512 * 512 * 3
RECORD_SIZE = IMAGE_SIZE * 2


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

        original_image = cv2.imread(original_file,
                                    cv2.IMREAD_COLOR)[:, :, ::-1]
        wire_frame_image = cv2.imread(wire_frame_file,
                                      cv2.IMREAD_COLOR)[:, :, ::-1]

        if original_image is None or wire_frame_image is None:
            raise Exception('can not read image {},{}'.format(original_file,
                                                              wire_frame_file))

        self.pack_file.write(np.ndarray.tobytes(original_image.reshape([-1])))
        self.pack_file.write(
            np.ndarray.tobytes(wire_frame_image.reshape([-1])))

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

class DataSetBuilder(object):
    """
    DataSetBuilder builds image dataset from given image directory.
    
    Specification of dataset wrote from this is follows.

    - Each data length is 512*512*3*2 byte as pair of original image and wire-framed image.
    - First of each data pair is original image.
    - Second of each data pair is wire-framed image.
    - all images are flattened with numpy's reshape method.
      - applied np.reshape(512*512*3), so aligns of data is RGBRGBRGB...
    - A data pack contains 1000 image pairs.

    DataSetBuilder requires directory structure of original and wire-framed images,
    one is what each directories should have same structure,
    one is what all file names of each pair in each directories should be same.

    """

    def __init__(self, original_dir, wire_frame_dir, out_dir, pack_size=1000):
        assert pack_size > 0

        self.original_dir = original_dir
        self.wire_frame_dir = wire_frame_dir
        self.out_dir = out_dir
        self.pack_size = pack_size

        self.__pack_name_format = 'image_pack_{}.bin'

    def build(self):
        if not os.path.exists(self.original_dir):
            raise Exception(
                'not found input directory: {}'.format(self.original_dir))

        if not os.path.exists(self.wire_frame_dir):
            raise Exception(
                'not found input directory: {}'.format(self.wire_frame_dir))

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        target_files = []
        for (root, _, files) in os.walk(self.original_dir):
            for f in files:
                target_files.append(f)

        file_num = len(target_files)
        (all_pack_num, reminder_pack) = divmod(file_num, self.pack_size)
        for pack_num in range(all_pack_num):
            self.__build_pack(pack_num + 1,
                              target_files[pack_num * self.pack_size:pack_num *
                                           self.pack_size + self.pack_size])

        # build pack if do not just divide size of target_files with pack_size.
        if reminder_pack != 0:
            self.__build_pack(all_pack_num + 1, target_files[-reminder_pack:])

    def __build_pack(self, pack_num, files):
        logging.debug('Start packing no:{}'.format(pack_num))
        output = open(
            os.path.join(self.out_dir,
                         self.__pack_name_format.format(pack_num)),
            mode='wb')

        image_pack = ImagePack(output)
        for f in files:
            origin = os.path.join(self.original_dir, f)
            wf = os.path.join(self.wire_frame_dir, f)
            image_pack.pack(origin, wf)

        output.close()
        logging.debug('Finish packing no:{}'.format(pack_num))


class DataSetReader(object):
    """
    DataSetReader provides read images from dataset, and randomized batch reading.

    Read datasets from this are random files and random record, so this do not provide
    method sequential read from dataset. So this class is only used for training!

    The format of image read from this class is follows.

    - shape of array is [read_size, 2, IMAGE_SIZE]
      IMAGE_SIZE = 512 * 512 * 3
    - the data type of all elements of array is 'float32'
      That mean of 1.0 in batch of images is 255 as byte,
      and mean of 0.0 in batch of imagesis 0 as byte.
    """

    def __init__(self, dataset_size=1000):
        self.dataset_files = []
        self.dataset_size = dataset_size
        self.file_list_size = 0

    def prepare(self, dataset_dir):
        """
        Prepare to read images from dataset.
        """
        for (root, _, files) in os.walk(dataset_dir):
            for f in files:
                self.dataset_files.append(os.path.join(root, f))

        self.file_list_size = len(self.dataset_files)
        random.seed()

    def read_batch(self, size):
        assert size > 0

        result = np.zeros([size, 2, IMAGE_SIZE], float)
        for i in range(size):
            file_index = random.randint(1, self.file_list_size)
            record_index = random.randint(1, self.dataset_size)
            target_file = self.dataset_files[file_index - 1]

            with open(target_file, 'rb') as fp:
                original, wire = ImagePack(fp).unpack(record_index)
                result[i] = [original / 255.0, wire / 255.0]

        return result
