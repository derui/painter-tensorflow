# coding: utf-8

import os


def load_exclude_names(excludes_dir):
    """load file names from the directory.

    @return list that files should exclude in process. Not contained extension of file
            in list.
    """
    excludes = []
    for r, _, files in os.walk(excludes_dir):
        names = [v for v, _ in map(lambda x: os.path.splitext(x), files)]
        excludes.extend(names)

    return set(excludes)


def walk_files(input_dir, exclude_files, per_yield_files):

    return_files = []
    ignored_files = 0
    for root, _, files in os.walk(input_dir):
        for f in files:
            n, _ = os.path.splitext(os.path.basename(f))
            if n in exclude_files:
                ignored_files += 1
                continue

            return_files.append((root, f))

            if len(return_files) >= per_yield_files:
                yield (return_files, ignored_files)

                return_files = []

    if len(return_files) > 0:
        yield (return_files, ignored_files)


def make_generic_processor(read_func, write_func, process):
    """make generic-image-processor funciton.

    generic image processor has three processes: reader, writer, and image processing.
    it accept only two arguments, such as input path and output path.

    @return new generic image processor
    """
    def func(in_path, out_path):
        item = read_func(in_path)
        item = process(item)
        write_func(item, out_path)

    return func
