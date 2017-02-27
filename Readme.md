# Line Art Painter for Tensorflow #

## Making datasets ##

1. collect images.
2. ``cd datasets``
3. ``python resize_fixed_size.py -d <output directory> <dataset directory>``
4. ``python extract_line_art.py -d <output directory> <output of resize_fixed_size.py>``
5. ``python packer.py -d <output directory> <output of resize_fixed_size.py> <output of extract_line_art.py>``

## Run training ##

1. Done making datasets before
2. ``python training.py --dataset_dir <output of packer.py>``

training.py has some options below.

- --dataset_dir
- --train_dir
- --nax_steps
- --full_trace
- --log\_device\_placement
