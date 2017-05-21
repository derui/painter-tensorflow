import os
import argparse
import threading
import numpy as np
import cv2 as cv
import concurrent.futures
import queue
from . import util

argparser = argparse.ArgumentParser(description='Extract edge layer of a color image')
argparser.add_argument('input_dir', type=str, help='the directory included images to extract edge layer')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-s', '--size', dest='size', type=int)
argparser.add_argument('-e', dest='excludes_dir', type=str)

args = argparser.parse_args()

neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
FIXED_SIZE = args.size


class Ignore(Exception):
    pass


gamma = 1.8

lookUpTable = np.zeros((256, 1), dtype='uint8')

for i in range(256):
    lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)


def extract_edge(rq, wq):

    try:
        img, path = rq.get_nowait()
    except queue.Empty:
        raise Ignore()
    rq.task_done()

    img = cv.LUT(img, lookUpTable)
    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_RGB2GRAY)

    if FIXED_SIZE is not None:
        img_diff_not = cv.adaptiveThreshold(img_diff_not, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 8)
        img_diff_not = util.resize_image(img_diff_not, FIXED_SIZE)
        img_dilate = cv.erode(img_diff_not, neiborhood8, iterations=1)
        img_dilate = cv.dilate(img_diff_not, neiborhood8, iterations=1)
        img_diff = cv.absdiff(img_diff_not, img_dilate)
        img_diff_not = cv.bitwise_not(img_diff)
        # img_diff_not = cv.adaptiveThreshold(img_diff_not, 255,
        #                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                 cv.THRESH_BINARY, 7, 8)

    img_diff_not = cv.cvtColor(img_diff_not, cv.COLOR_GRAY2RGB)

    wq.put((img_diff_not, path))


def read_image(path):
    img = cv.imread(path)
    if img is None:
        raise Exception("OpenCV can not load %s" % (path))
    return img


def write_image(info, out_dir):
    img, path = info

    dirname, fname = os.path.split(os.path.abspath(path))
    fname, ext = os.path.splitext(fname)
    d = os.path.join(out_dir, fname[0:2])
    if not os.path.exists(d):
        os.makedirs(d, 0o755, exist_ok=True)

    writefname = os.path.join(d, "{}{}".format(fname, ext))

    cv.imwrite(writefname, img)


excludes = []
if args.excludes_dir is not None:
    for r, _, files in os.walk(args.excludes_dir):
        excludes.extend(files)

path_queue = util.make_sequential_queue(args.input_dir, excludes)

image_queue = queue.LifoQueue(300)
write_queue = queue.LifoQueue(300)

reader_event = threading.Event()
readerExecutor = concurrent.futures.ThreadPoolExecutor(8)
read_thread = [
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event)),
    readerExecutor.submit(util.sequential_read_dir(path_queue, image_queue, read_image, reader_event))
]

writer_event = threading.Event()
writerExecutor = concurrent.futures.ThreadPoolExecutor(8)
write_thread = [
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event)),
    writerExecutor.submit(util.queue_writer(args.out_dir, write_queue, write_image, writer_event))
]

executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def reader_process(event):
    num = 0

    while not event.is_set():
        futures = []
        for _ in range(14):
            futures.append(executor.submit(extract_edge, image_queue, write_queue))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Ignore:
                pass
            except Exception as exc:
                print('exception: %s' % exc)
            else:
                num += 1
                if num % 100 == 0:
                    print('Completed {} items'.format(num))


processor_event = threading.Event()
main_processor = executor.submit(reader_process, processor_event)

path_queue.join()
print('Finish reading all paths')

image_queue.join()
reader_event.set()
readerExecutor.shutdown()
print('Finish to read all pathes from queue')

write_queue.join()
writer_event.set()
writerExecutor.shutdown()
print('Finish to write all images converted')

processor_event.set()
