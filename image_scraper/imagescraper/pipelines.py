# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import numpy as np
import os
import re
from scrapy.exceptions import DropItem
from scrapy.pipelines.files import FilesPipeline, FileException
import cv2 as cv

ITEM_MIN_SIZES = {'w': 300, 'h': 300}
IGNORE_TAGS = [
    re.compile('comic'),
    re.compile('monochrome'),
    re.compile('translation request'),
    re.compile('greyscale'),
    re.compile('text'),
    re.compile('commentary'),
    re.compile('4coma'),
    re.compile('^photo$'),
    re.compile('^real$'),
]


def _include_ignoreable_tags(tags):
    for tag in tags:
        for pattern in IGNORE_TAGS:
            if pattern.match(tag):
                return True

    return False


def _inference_line_art(img):
    img_tmp = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    img_dilate = cv.dilate(img_tmp, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img_tmp, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff = cv.absdiff(img_tmp, img_diff_not)
    accuracy = np.mean(img_diff < 10)

    return accuracy > 0.95


def _valid_constraint(img):
    w, h, _ = img.shape

    if w < ITEM_MIN_SIZES['w']:
        return False

    if h < ITEM_MIN_SIZES['h']:
        return False

    return True


def _write_ignore_file(basedir, path):
    basedir = os.path.join(basedir, 'excluded')
    filename, _ = os.path.splitext(os.path.basename(path))
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    tagfile = os.path.join(basedir, filename)
    with open(tagfile, 'w'):
        pass


class ImageScraperPipeline(FilesPipeline):
    def __init__(self, store_uri, download_func=None, settings=None):
        super(ImageScraperPipeline, self).__init__(store_uri, download_func,
                                                   settings)

    # def get_media_requests(self, item, info):
    #     headers = item['response'].headers.copy()
    #     headers['referer'] = item['response'].url
    #     return [scrapy.Request(item.get('file_urls')[0], headers=headers)]

    def item_completed(self, results, item, info):

        ok, x = results[0]
        if not ok:
            raise FileException("Item not contains")

        item['files'] = [x['path']]
        tags = item['tags']
        item['tags'] = {}

        self._save_tags(x['path'], tags)
        self._constraint_image(x['path'])

        return item

    def _save_tags(self, path, tags):
        basedir = os.path.join(self.store.basedir, 'tags')
        filename, _ = os.path.splitext(os.path.basename(path))

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        tagfile = os.path.join(basedir, filename + '.tsv')

        with open(tagfile, "w") as f:
            for tag in tags:
                f.write(tag + "\n")

        if _include_ignoreable_tags(tags):
            _write_ignore_file(self.store.basedir, path)

    def _constraint_image(self, path):

        try:
            img = cv.imread(os.path.join(self.store.basedir, path))

            if img is None:
                raise DropItem('Item is not readable')

            if not _valid_constraint(img):
                raise DropItem('Item is not valid size')

            if _inference_line_art(img):
                raise DropItem('Item is likely line art')

            cv.imwrite(os.path.join(self.store.basedir, path), img)

        except DropItem as e:
            _write_ignore_file(self.store.basedir, path)
