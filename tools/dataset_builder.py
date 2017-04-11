# -*- coding: utf-8 -*-

import argparse
from . import dataset as ip
import logging
import os
import cv2


class DataSetBuilder(object):
    """
    DataSetBuilder builds image dataset from given image directory.
    
    Specification of dataset wrote from this is follows.

    - Each data length is 512*512*3*2 byte as pair of original image and wire-framed image.
    - First of each data pair is original image.
    - Second of each data pair is wire-framed image.
    - all images are flattened with numpy's reshape method.
      - applied np.reshape(512*512*3), so aligns of data is RGBRGBRGB...
    - A data pack contains 5000 image pairs.

    DataSetBuilder requires directory structure of original and wire-framed images,
    one is what each directories should have same structure,
    one is what all file names of each pair in each directories should be same.

    """

    def __init__(self, original_dir, wire_frame_dir, out_dir, pack_size=5000):
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

        target_files = set()
        for (root, _, files) in os.walk(self.original_dir):
            for f in files:
                _, prefix = os.path.split(root)
                target_files.add(os.path.join(prefix, f))

        diff_files = set()
        for (root, _, files) in os.walk(self.wire_frame_dir):
            for f in files:
                _, prefix = os.path.split(root)
                diff_files.add(os.path.join(prefix, f))

        target_files = list(target_files.intersection(diff_files))

        file_num = len(target_files)
        (all_pack_num, reminder_pack) = divmod(file_num, self.pack_size)

        for pack_num in range(all_pack_num):
            self.__build_pack(pack_num + 1,
                              target_files[pack_num * self.pack_size:pack_num *
                                           self.pack_size + self.pack_size])

        # build pack if do not just divide size of target_files with pack_size.
        if reminder_pack != 0:
            self.__build_pack(all_pack_num + 1, target_files[-reminder_pack:])

    def __read_images(self, f):
        original_file = os.path.join(self.original_dir, f)
        line_art_file = os.path.join(self.wire_frame_dir, f)

        if not os.path.exists(original_file):
            raise Exception(
                'not found original file: {}'.format(original_file))

        if not os.path.exists(line_art_file):
            raise Exception(
                'not found wire frame file: {}'.format(line_art_file))

        original_image = cv2.imread(original_file, cv2.IMREAD_COLOR)
        line_art_image = cv2.imread(line_art_file, cv2.IMREAD_COLOR)

        if original_image is None or line_art_image is None:
            raise Exception('can not read image {},{}'.format(original_file,
                                                              line_art_file))

        return original_image, line_art_image

    def __build_pack(self, pack_num, files):
        logging.debug('Start packing no:{}'.format(pack_num))
        output = open(
            os.path.join(self.out_dir,
                         self.__pack_name_format.format(pack_num)),
            mode='wb')

        image_pack = ip.ImagePack(output)
        images = [self.__read_images(f) for f in files]

        for (origin, la) in images:
            image_pack.pack(origin, la)

        output.close()
        logging.debug('Finish packing no:{}'.format(pack_num))


argparser = argparse.ArgumentParser(
    description='Packing images for traininig data set')
argparser.add_argument(
    'original_dir', type=str, help='the directory included original images')
argparser.add_argument(
    'wire_frame_dir',
    type=str,
    help='the directory included extracted edge layer')
argparser.add_argument('-d', dest='out_dir', type=str)

args = argparser.parse_args()

logging.basicConfig(level=logging.DEBUG)
# the directory to output
out_dir = '.' if args.out_dir is None else args.out_dir

builder = DataSetBuilder(args.original_dir, args.wire_frame_dir, out_dir)

builder.build()
