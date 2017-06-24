# -*- coding: utf-8 -*-
import os
from imagescraper.items import ImageScraperItem
import scrapy

IGNORE_TAGS = ['comic', 'monochrome', 'tagme']


def _include_ignoreable_tags(tags):
    for tag in tags:
        if tag in IGNORE_TAGS:
            return True

    return False


class SafebooruSpider(scrapy.Spider):
    name = "safebooru"
    allowed_domains = ["safebooru.org"]

    def start_requests(self):
        initial_offset = 0 if 'offset' not in self.state else self.state['offset']
        for offset in range(initial_offset, initial_offset+1000):
            self.state['offset'] = offset
            print('http://safebooru.org/index.php?page=dapi&s=post&q=index&pid={}'.format(offset))

            yield self.make_requests_from_url(
                'http://safebooru.org/index.php?page=dapi&s=post&q=index&pid={}'.
                format(offset))

    def parse(self, response):
        posts = response.xpath('//post')

        tag_map = {}
        file_urls = []
        for post in posts:
            file_url = 'http:' + post.xpath('@file_url').extract_first()
            tags = post.xpath('@tags').extract_first().split(' ')
            tags = list(filter(lambda x: x != '', tags))

            if self._ignore_tags(tags):
                continue

            if file_url is not None and not self.__should_ignore(file_url):
                file_urls.append(file_url)
                tag_map[file_url] = tags

        item = ImageScraperItem(
            tags=tag_map,
            file_urls=file_urls,
            files=[]
        )

        yield item
    
    def _ignore_tags(self, tags):
        """return if tags contains some tags should be ignore
        """

        if _include_ignoreable_tags(tags):
            return True

        return False

    def __should_ignore(self, url):
        p = os.path.basename(url)
        _, ext = os.path.splitext(p)

        ignoreable_exts = ['.gif']
        return ext in ignoreable_exts
