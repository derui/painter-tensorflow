# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os
import scrapy
from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem
import cv2 as cv
import numpy as np

ITEM_MIN_SIZES = {'w': 300, 'h': 300}


def _valid_constraint(img):
    w, h = img.shape

    if w < ITEM_MIN_SIZES['w']:
        return False

    if h < ITEM_MIN_SIZES['h']:
        return False

    return True


def _is_line_art(img):
    """Detect color image"""
    grayed = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    r_diff = np.abs(img[::, ::, 2] - grayed)
    g_diff = np.abs(img[::, ::, 1] - grayed)
    b_diff = np.abs(img[::, ::, 0] - grayed)

    thresholds = 25
    likelihood = 0.8
    diffs = np.array(
        [r_diff < thresholds, g_diff < thresholds, b_diff < thresholds])

    return np.alltrue(diffs > likelihood)


class ImageScraperPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        headers = item['response'].headers.copy()
        headers['referer'] = item['response'].url
        return [scrapy.Request(item.get('file_urls')[0], headers=headers)]

    def item_completed(self, results, item, info):
        ok, x = results[0]

        if not ok or not x['path']:
            raise DropItem('Item contains no images')

        path = os.path.join(self.store.basedir, x['path'])
        img = cv.imread(path, cv.IMREAD_COLOR)

        if img is None or _is_line_art(img):
            os.unlink(path)
            raise DropItem('Item is likely as line art')

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if not _valid_constraint(img):
            os.unlink(path)
            raise DropItem('Item is illegal size by image constraint')

        item['files'] = [x['path']]
        return item
