# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import numpy as np
import os
import scrapy
from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem
import cv2 as cv

ITEM_MIN_SIZES = {'w': 300, 'h': 300}
IGNORE_TAGS = ['comic', 'monochrome']


def _inference_line_art(img):
    neiborhood8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    img_dilate = cv.dilate(img, neiborhood8, iterations=1)
    img_diff = cv.absdiff(img, img_dilate)
    img_diff_not = cv.bitwise_not(img_diff)
    img_diff = cv.absdiff(img, img_diff_not)
    accuracy = np.mean(img_diff < 10)

    return accuracy > 0.95


def _valid_constraint(img):
    w, h = img.shape

    if w < ITEM_MIN_SIZES['w']:
        return False

    if h < ITEM_MIN_SIZES['h']:
        return False

    return True


def _include_ignoreable_tags(tags):
    for tag in tags:
        if tag in IGNORE_TAGS:
            return True

    return False


class ImageScraperPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        headers = item['response'].headers.copy()
        headers['referer'] = item['response'].url
        return [scrapy.Request(item.get('file_urls')[0], headers=headers)]

    def item_completed(self, results, item, info):
        ok, x = results[0]

        tags = item.get('tags')
        item['tags'] = []
        path = os.path.join(self.store.basedir, x['path'])

        if not ok or not x['path']:
            raise DropItem('Item contains no images')

        if _include_ignoreable_tags(tags):
            self._ignore_file(path)
            raise DropItem('Item is posted had any ignoreable tag')

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)

        if img is None:
            self._ignore_file(path)
            raise DropItem('Item is not readable')

        if not _valid_constraint(img):
            self._ignore_file(path)
            raise DropItem('Item is illegal size by image constraint')

        if _inference_line_art(img):
            self._ignore_file(path)
            raise DropItem('Item is line art probably {}'.format(path))

        self._save_tags(x['path'], tags)

        item['files'] = [x['path']]
        return item

    def _ignore_file(self, path):
        basedir = os.path.join(self.store.basedir, 'excluded')
        filename, _ = os.path.splitext(os.path.basename(path))
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        tagfile = os.path.join(basedir, filename)
        with open(tagfile, 'w'):
            pass

    def _save_tags(self, path, tags):
        basedir = os.path.join(self.store.basedir, 'tags')
        filename, _ = os.path.splitext(os.path.basename(path))

        if not os.path.exists(basedir):
            os.makedirs(basedir)

        tagfile = os.path.join(basedir, filename + '.tsv')

        with open(tagfile, "w") as f:
            for tag in tags:
                f.write(tag + "\n")
