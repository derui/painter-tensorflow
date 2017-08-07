import os
import argparse
import concurrent.futures
from datetime import datetime
from ....tflib import util
import re
import hashlib


def read_tag(path):

    tags = []
    with open(path) as f:
        tags = list(map(lambda x: x.strip(), f.readlines()))

    return tags


def write_tag(tag, out_path):

    dirname, fname = os.path.split(os.path.abspath(out_path))
    if not os.path.exists(dirname):
        os.makedirs(dirname, 0o755, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(','.join([str(v) for v in tag]))


def make_process(all_tags, origin):
    def process(tags):
        zeros = origin.copy()

        for tag in tags:
            if tag in all_tags:
                zeros[all_tags[tag]] = 1

        return zeros

    return process


def to_out_path(out_dir, f):
    name, _ = os.path.splitext(f)
    return os.path.join(out_dir, f[:2], "{}.csv".format(name))


def is_unreliable_tag(tag):
    """check unreliable tags from tag set
    """

    MATCHERS = [
        lambda x: x == "...",
        lambda x: not re.match("^[a-zA-Z]", x)
    ]

    for matcher in MATCHERS:
        if matcher(tag):
            return True

    return False


def normalize(tag):
    tag = tag.strip()
    tag = tag.replace(" ", "_")

    return tag


def main(args, excludes):
    num = 0
    tag_set = set()
    for files, ignored_files in util.walk_files(args.input_dir, excludes, 1000):
        for root, f in files:
            for tag in read_tag(os.path.join(root, f)):
                tag = normalize(tag)
                if is_unreliable_tag(tag):
                    continue

                if tag not in tag_set:
                    tag_set.append(tag)

        num += len(files)
        print("{}: merged files {}".format(datetime.now(), num))

    def hash_key(key):
        s = hashlib.sha256()
        s.update(key.encode("utf-8"))
        return s.hexdigest()

    tag_map = {hash_key(v): v for v in tag_set}
    image_processor = util.make_generic_processor(read_tag, write_tag,
                                                  make_process(tag_map))

    num = 0
    for files, ignored_files in util.walk_files(args.input_dir, excludes, 100):
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
            futures = []
            for root, f in files:
                futures.append(
                    e.submit(image_processor,
                             os.path.join(root, f),
                             to_out_path(args.out_dir, f)))

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print('exception: %s' % exc)

            num += 100
            print('{}: Completed {} items, {} ignored.'.format(
                datetime.now(), num, ignored_files))

    with open(args.mapping_file, "w") as f:
        f.write(','.join([x for x, _ in sorted_by_count_list]))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='merge tags and make sparse data')
    argparser.add_argument(
        'input_dir',
        type=str,
        help='the directory of image to resize and crop to fixed size')
    argparser.add_argument('-d', dest='out_dir', type=str, required=True)
    argparser.add_argument('-o', dest='mapping_file', type=str, required=True)
    argparser.add_argument('-e', dest='excludes_dir', type=str)

    args = argparser.parse_args()

    excludes = []
    if args.excludes_dir is not None:
        excludes = util.load_exclude_names(args.excludes_dir)

    main(args, excludes)
