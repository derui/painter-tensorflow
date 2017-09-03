import argparse
from datetime import datetime
from . import util
from tflib import util as tfutil
import pathlib


def normalize_tags(tags):
    ret = []

    for tag in tags:
        ret.extend(util.normalize(tag))

    return ret


def read_tag(path):

    tags = []
    with open(path) as f:
        tags = list(map(lambda x: x.strip(), f.readlines()))
        tmp_tag_list = list(filter(lambda x: not util.is_unreliable_tag(x), tags))
        tmp_tag_list = normalize_tags(tmp_tag_list)

    return tmp_tag_list


def write_tag(tags, out_path, vocab):
    with open(str(out_path), "w") as f:
        tmp_tag_list = list(map(vocab.get, tags))

        tmp_tag_list = list(map(lambda x: str(x)+"\n", tmp_tag_list))
        f.writelines(tmp_tag_list)


def make_process(all_tags, origin):
    return None


def main(args, excludes):
    vocab = util.Vocabulary()
    num = 0
    for files, ignored_files in tfutil.walk_files(args.input_dir, excludes,
                                                  1000):
        for root, f in files:
            for tag in read_tag(str(pathlib.Path(root) / f)):
                vocab.append(tag)

        num += len(files)
        print("{}: merged files {}".format(datetime.now(), num))

    vocab.write(str(pathlib.Path(args.out_dir) / args.out_file))

    vocab.trim(200)
    vocab.freeze()

    vocab.write(str(pathlib.Path(args.out_dir) / ("small_" + args.out_file)))

    num = 0
    for files, ignored_files in tfutil.walk_files(args.input_dir, excludes,
                                                  1000):
        for root, f in files:
            tags = read_tag(str(pathlib.Path(root) / f))

            out_path = pathlib.Path(args.out_dir) / "tags" / f
            write_tag(tags, out_path, vocab)

        num += len(files)
        print("{}: write normalized tags {}".format(datetime.now(), num))


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='make vocabulary of tags')
    argparser.add_argument(
        'input_dir',
        type=str,
        help='the directory of image to resize and crop to fixed size')
    argparser.add_argument('-d', dest='out_dir', type=str, required=True)
    argparser.add_argument('-o', dest='out_file', type=str, required=True)
    argparser.add_argument('-e', dest='excludes_dir', type=str)

    args = argparser.parse_args()

    excludes = []
    if args.excludes_dir is not None:
        excludes = tfutil.load_exclude_names(args.excludes_dir)

    path = pathlib.Path(args.out_dir)
    if not path.exists():
        path.mkdir()

    if not (path / "tags").exists():
        (path / "tags").mkdir()

    main(args, excludes)
