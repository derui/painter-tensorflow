# Line Art Painter for Tensorflow #

## Making datasets ##

1. collect images.
2. ``cd datasets``
3. ``python resize_fixed_size.py -d <output directory> <dataset directory>``
4. ``python extract_line_art.py -d <output directory> <output of resize_fixed_size.py>``
5. ``python packer.py -d <output directory> <output of resize_fixed_size.py> <output of extract_line_art.py>``

## Run training for Line-Art painter with WGAN ##

1. Done making datasets before
2. ``python -m line_art_painter.training_wgan --dataset_dir <output of packer.py> --train_dir <log of training>``

training.py has some options below.

- --dataset_dir
- --train_dir
- --nax_steps
- --full_trace
- --lambda_
- --learning_rate
- --beta1
- --critic_step
- --batch_size
- --log\_device\_placement
