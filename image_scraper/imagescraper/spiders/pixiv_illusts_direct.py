# -*- coding: utf-8 -*-
from imagescraper.items import ImageScraperItem
import scrapy
import random


class PixivIllustsDirectSpider(scrapy.Spider):
    name = "pixiv_illusts"
    allowed_domains = ["www.pixiv.net"]

    downloaded_id = []

    def start_requests(self):
        for i in range(500000):
            illust_id = random.randint(10000000, 70000000)
            while True:
                if illust_id not in self.downloaded_id:
                    break
                illust_id = random.randint(10000000, 70000000)

            self.downloaded_id.append(illust_id)
            yield self.make_requests_from_url(
                'http://www.pixiv.net/member_illust.php?mode=medium&illust_id={}'.
                format(illust_id))

    def parse(self, response):
        img = response.css('div.img-container').xpath('.//img')
        img_url = img.xpath('@src').extract_first()

        if img_url is not None:
            item = ImageScraperItem(
                file_urls=[img_url], files=[], response=response)

            yield item
