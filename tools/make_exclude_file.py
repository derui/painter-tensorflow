import argparse
import pathlib

argparser = argparse.ArgumentParser(description='Make file to detect file is target to exclude')
argparser.add_argument('input_dir', type=str, help='the directory included names to exclude')
argparser.add_argument('-f', dest='out_file', type=str, required=True)

args = argparser.parse_args()


if __name__ == "__main__":

    excludes = []
    path = pathlib.Path(args.input_dir)
    for f in path.iterdir():
        if f.is_dir():
            continue
        excludes.append(f.name + "\n")

    with open(args.out_file, "w") as f:
        f.writelines(excludes)
