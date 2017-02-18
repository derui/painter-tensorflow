# -*- coding: utf-8 -*-
from image_scraper.items import ImageScraperItem
import scrapy
import urllib


class PixivIllustsSpider(scrapy.Spider):
    name = "pixiv_illusts"
    allowed_domains = ["www.pixiv.net"]
    start_urls = [
        'http://www.pixiv.net/search.php?word=%E8%89%A6%E3%81%93%E3%82%8C&type=illust',
        'http://www.pixiv.net/search.php?word=%E6%9D%B1%E6%96%B9&type=illust',
    ]

    def parse(self, response):
        """Parse pixiv's search page for illusts.
        """

        member_pages = response.css(
            "ul._image-items.autopagerize_page_element"
        ).xpath(
            'li[contains(@class, "image-item")]/a[contains(@class, "work")]')
        for member_page in member_pages:
            page_url = member_page.xpath("@href").extract_first()
            member_url = urllib.parse.urljoin(response.url, page_url)
            yield scrapy.Request(
                member_url,
                callback=self.parse_page,
                headers={'referer': response.url})

        # parse next page if it exists.

        next_page = response.css('span.next').xpath('./a[@rel="next"]')
        next_page_url = next_page.xpath('@href').extract_first()
        yield scrapy.Request(
            urllib.parse.urljoin(response.url, next_page_url),
            callback=self.parse)

    def parse_page(self, response):
        img = response.css('div.img-container').xpath('.//img')
        img_url = img.xpath('@src').extract_first()

        item = ImageScraperItem(
            file_urls=[img_url], files=[], response=response)

        yield item
