#!/bin/bash

MAX_DOCUMENT_LENGTH=1091
DIR=datasets/dnn_upscaler
SUFFIX=$1

echo "Start resizing..."
cd ..

# mkdir -p $DIR
# python -m line_art_painter.lib.tools.resize_fix_size -s 512 -d $DIR/resized${SUFFIX} -e images/excluded images/full

echo "Start packing to tfrecord"
python -m dnn_upscaler.lib.tools.convert_to_tfrecord --image_dir $DIR/resized${SUFFIX} --out_dir $DIR/packed${SUFFIX}
