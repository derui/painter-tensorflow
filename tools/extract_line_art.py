import boto3
import argparse
import pathlib
from datetime import datetime
import numpy as np
import cv2 as cv
import util
import concurrent.futures

argparser = argparse.ArgumentParser(description='Extract edge layer of a color image')
argparser.add_argument('prefix', type=str, help='the prefix of images to extract edge layer')
argparser.add_argument('-b', dest='bucket', type=str, required=True)
argparser.add_argument('-d', dest='output_key_prefix', type=str, required=True)
argparser.add_argument('-e', dest='exclude_file_key', type=str, required=True)
argparser.add_argument('-p', dest='parallel', type=int, default=8, help="number of task in parallel")

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)


def extract_edge(img):

    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_RGB2GRAY)

    return img_diff_not


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


def process(bucket, output_prefix, content):
    session = boto3.session.Session()
    s3 = session.client("s3")

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
        img = extract_edge(img)
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for page in iterator:
            if 'Contents' not in page:
                continue

            for content in page['Contents']:
                if not should_process(content['Key']):
                    ignored += 1
                    continue

                executor.submit(process, args.bucket, args.output_key_prefix, content)

            num += len(page['Contents'])

    print('{}: Completed {} items, {} ignored'.format(datetime.now(), num, ignored))
