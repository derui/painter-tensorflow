# -*- coding: utf-8 -*-

import argparse
from . import dataset as ip

argparser = argparse.ArgumentParser(
    description='Packing images for traininig data set')
argparser.add_argument(
    'original_dir',
    type=str,
    help='the directory included original images')
argparser.add_argument(
    'wire_frame_dir',
    type=str,
    help='the directory included extracted edge layer')
argparser.add_argument('-d', dest='out_dir', type=str)

args = argparser.parse_args()

# the directory to output
out_dir = '.' if args.out_dir is None else args.out_dir

builder = ip.DataSetBuilder(args.original_dir, args.wire_frame_dir,
                            out_dir)

builder.build()
