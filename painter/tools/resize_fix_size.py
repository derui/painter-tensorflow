import os
import argparse
import cv2 as cv
import concurrent.futures
import queue
import threading
from . import util

argparser = argparse.ArgumentParser(description='Resize image to fixed size')
argparser.add_argument('input_dir', type=str, help='the directory of image to resize and crop to fixed size')
argparser.add_argument('-d', dest='out_dir', type=str, required=True)
argparser.add_argument('-e', dest='excludes_dir', type=str)
argparser.add_argument('-s', '--size', dest='size', type=int)

args = argparser.parse_args()


class Ignore(Exception):
    pass


def process(rq, wq):

    img, path = rq.get()
    rq.task_done()

    img = util.resize_image(img, FIXED_SIZE)

    wq.put((img, path))


FIXED_SIZE = 512 if args.size is None else args.size


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

    writefname = os.path.join(d, "{}{}".format(fname, '.png'))

    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

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
            futures.append(executor.submit(process, image_queue, write_queue))

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

image_queue.join()
reader_event.set()
print('Finish to read all pathes from queue')

write_queue.join()
writer_event.set()
writerExecutor.shutdown()

processor_event.set()
