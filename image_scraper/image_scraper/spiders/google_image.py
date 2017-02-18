# -*- coding: utf-8 -*-
import scrapy


class GoogleImageSpider(scrapy.Spider):
    name = "google_image"
    allowed_domains = ["www.pixiv.com"]
    start_urls = ['http://www.pixiv.net/search.php?word=%E8%89%A6%E3%81%93%E3%82%8C&type=illust']

    def parse(self, response):
        # let's only gather Time U.S. magazine covers
        url = response.css("div.refineCol ul li").xpath("a[contains(., 'TIME U.S.')]")
        yield scrapy.Request(url.xpath("@href").extract_first(), self.parse_page)

    def parse_page(self, response):
        pass

