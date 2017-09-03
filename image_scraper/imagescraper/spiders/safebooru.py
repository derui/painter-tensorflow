# -*- coding: utf-8 -*-
import os
from imagescraper.items import ImageScraperItem
import scrapy


class SafebooruSpider(scrapy.Spider):
    name = "safebooru"
    allowed_domains = ["safebooru.org"]

    def __init__(self, offset=None, *args, **kwargs):
        self.offset = int(offset)
        super(SafebooruSpider, self).__init__(*args, **kwargs)

    def start_requests(self):
        initial_offset = 0 if 'offset' not in self.state else self.state['offset']
        for offset in range(initial_offset, initial_offset+self.offset):
            self.state['offset'] = offset
            print('http://safebooru.org/index.php?page=dapi&s=post&q=index&pid={}'.format(offset))

            yield self.make_requests_from_url(
                'http://safebooru.org/index.php?page=dapi&s=post&q=index&pid={}'.
                format(offset))

    def parse(self, response):
        posts = response.xpath('//post')

        for post in posts:
            file_url = 'http:' + post.xpath('@file_url').extract_first()
            tags = post.xpath('@tags').extract_first().split(' ')
            tags = list(filter(lambda x: x != '', tags))

            if file_url is not None and not self.__should_ignore(file_url):

                item = ImageScraperItem(
                    tags=tags,
                    file_urls=[file_url],
                    files=[]
                )

                yield item

    def __should_ignore(self, url):
        p = os.path.basename(url)
        _, ext = os.path.splitext(p)

        ignoreable_exts = ['.gif']
        return ext in ignoreable_exts
