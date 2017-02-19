# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.pipelines.files import FilesPipeline


class ImageScraperPipeline(FilesPipeline):
    def get_media_requests(self, item, info):
        headers = item['response'].headers.copy()
        headers['referer'] = item['response'].url
        return [scrapy.Request(item.get('file_urls')[0], headers=headers)]
