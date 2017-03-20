# -*- coding: utf-8 -*-

import argparse
from . import dataset as ip
import logging
import os


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
                _, prefix = os.path.split(root)
                target_files.append((prefix, f))

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

        image_pack = ip.ImagePack(output)
        for prefix, f in files:
            origin = os.path.join(self.original_dir, prefix, f)
            wf = os.path.join(self.wire_frame_dir, prefix, f)
            image_pack.pack(origin, wf)

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

# the directory to output
out_dir = '.' if args.out_dir is None else args.out_dir

builder = ip.DataSetBuilder(args.original_dir, args.wire_frame_dir, out_dir)

builder.build()
