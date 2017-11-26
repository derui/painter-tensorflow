import boto3
import argparse
import cv2 as cv
import pathlib
from datetime import datetime
import util

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument('prefix', type=str, help='the prefix of images to resize and crop to fixed size')
argparser.add_argument('-b', dest='bucket', type=str, required=True)
argparser.add_argument('-d', dest='output_key_prefix', type=str, required=True)
argparser.add_argument('-e', dest='exclude_file_key', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)
argparser.add_argument('--crop', action='store_true')

args = argparser.parse_args()

FIXED_SIZE = 512 if args.size is None else args.size


def read_image(downloader, key):
    path = downloader(key, "/tmp")

    img = cv.imread(str(path))
    if img is None:
        raise Exception("OpenCV can not load {}".format(path))
    return img, path


def write_image(uploader, path, img):

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    out_path = path.with_suffix('.png')
    cv.imwrite(str(out_path), img)

    uploader(path.with_suffix('.png').name, out_path)
    pathlib.Path(out_path).unlink()


def process(s3, bucket, output_prefix, content):
    def downloader(key, output_dir):
        path = pathlib.PurePath('/tmp') / pathlib.PurePath(key).name
        s3.download_file(bucket, key, str(path))

        return path

    def uploader(name, path):
        key = pathlib.PurePath(output_prefix) / name

        with open(str(path), 'rb') as f:
            s3.upload_fileobj(f, bucket, str(key))

    try:
        img, path = read_image(downloader, content['Key'])
        img = util.resize_image(img, FIXED_SIZE, args.crop)
        write_image(uploader, path, img)

        if pathlib.Path(path).exists():
            pathlib.Path(path).unlink()

    except Exception as e:
        print(content['Key'], e)


if __name__ == "__main__":

    # Create a client
    client = boto3.client('s3')

    should_process = util.get_obj_exclusion_detector(client, args.bucket, args.exclude_file_key)
    iterator = util.get_obj_iterator(client, args.bucket, args.prefix)

    print('{}: Start processing'.format(datetime.now()))

    num = 0
    ignored = 0
    for page in iterator:
        if 'Contents' not in page:
            continue

        for content in page['Contents']:
            if not should_process(content['Key']):
                ignored += 1
                continue

            process(client, args.bucket, args.output_key_prefix, content)
        num += len(page['Contents'])
        print('{}: Completed {} items, {} ignored'.format(datetime.now(), num, ignored))
