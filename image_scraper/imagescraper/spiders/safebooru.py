# -*- coding: utf-8 -*-
from imagescraper.items import ImageScraperItem
import scrapy


class SafebooruSpider(scrapy.Spider):
    name = "safebooru"
    allowed_domains = ["safebooru.org"]

    def start_requests(self):
        for offset in range(1500, 2500):

            yield self.make_requests_from_url(
                'http://safebooru.org/index.php?page=dapi&s=post&q=index&pid={}'.
                format(offset))

    def parse(self, response):
        posts = response.xpath('//post')
        for post in posts:
            file_url = 'http:' + post.xpath('@file_url').extract_first()
            tags = post.xpath('@tags').extract_first().split(' ')
            tags = list(filter(lambda x: x != '', tags))

            if file_url is not None:
                item = ImageScraperItem(
                    tags=tags,
                    checksums=[],
                    file_urls=[file_url],
                    files=[],
                    response=response)

                yield item
