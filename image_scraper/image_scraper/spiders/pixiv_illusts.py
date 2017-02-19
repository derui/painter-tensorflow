# -*- coding: utf-8 -*-
from image_scraper.items import ImageScraperItem
import scrapy
import urllib
import logging


class PixivIllustsSpider(scrapy.Spider):
    name = "pixiv_illusts"
    allowed_domains = ["www.pixiv.net"]
    start_urls = [
        'http://www.pixiv.net/search.php?word=%E8%89%A6%E3%81%93%E3%82%8C&type=illust',
        'http://www.pixiv.net/search.php?word=%E6%9D%B1%E6%96%B9&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=%E3%82%AA%E3%83%AA%E3%82%B8%E3%83%8A%E3%83%AB&type=illust',
        'http://www.pixiv.net/search.php?word=%E5%89%B5%E4%BD%9C&type=illust',
        'http://www.pixiv.net/search.php?word=%E9%AD%94%E6%B3%95%E5%B0%91%E5%A5%B3%E3%81%BE%E3%81%A9%E3%81%8B%E2%98%86%E3%83%9E%E3%82%AE%E3%82%AB&type=illust',
        'http://www.pixiv.net/search.php?word=%E3%83%98%E3%82%BF%E3%83%AA%E3%82%A2&type=illust',
        'http://www.pixiv.net/search.php?word=%E5%A5%B3%E3%81%AE%E5%AD%90&type=illust',
        'http://www.pixiv.net/search.php?word=%E3%82%AA%E3%83%AA%E3%82%AD%E3%83%A3%E3%83%A9&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=VOCALOID&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=%E3%81%AA%E3%81%AB%E3%81%93%E3%82%8C%E3%81%8B%E3%82%8F%E3%81%84%E3%81%84&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=%E3%83%9D%E3%82%B1%E3%83%A2%E3%83%B3&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=%E5%88%9D%E9%9F%B3%E3%83%9F%E3%82%AF&type=illust',
        'http://www.pixiv.net/search.php?s_mode=s_tag&word=%E3%82%A2%E3%82%A4%E3%83%89%E3%83%AB%E3%83%9E%E3%82%B9%E3%82%BF%E3%83%BC%E3%82%B7%E3%83%B3%E3%83%87%E3%83%AC%E3%83%A9%E3%82%AC%E3%83%BC%E3%83%AB%E3%82%BA&type=illust',
    ]

    searched_tags = []

    def parse(self, response):
        """Parse pixiv's search page for illusts.
        """

        # parse next page if it exists.
        member_pages = response.css(
            "ul._image-items.autopagerize_page_element"
        ).xpath(
            'li[contains(@class, "image-item")]/a[contains(@class, "work")]')
        for member_page in member_pages:
            page_url = member_page.xpath("@href").extract_first()
            member_url = urllib.parse.urljoin(response.url, page_url)
            yield scrapy.Request(
                member_url,
                callback=self.parse_member_page,
                headers={'referer': response.url})

        next_page = response.css('span.next').xpath('./a[@rel="next"]')
        next_page_url = next_page.xpath('@href').extract_first()

        if next_page_url is not None:
            yield scrapy.Request(
                urllib.parse.urljoin(response.url, next_page_url),
                callback=self.parse_page)

        for tag in self.__related_tags(response):
            yield scrapy.Request(
                '{}&type=illust'.format(
                    urllib.parse.urljoin(response.url, tag)),
                callback=self.parse_page)

    def parse_page(self, response):
        """Parse pixiv's search page for illusts.
        """

        # parse next page if it exists.
        member_pages = response.css(
            "ul._image-items.autopagerize_page_element"
        ).xpath(
            'li[contains(@class, "image-item")]/a[contains(@class, "work")]')
        for member_page in member_pages:
            page_url = member_page.xpath("@href").extract_first()
            member_url = urllib.parse.urljoin(response.url, page_url)
            yield scrapy.Request(
                member_url,
                callback=self.parse_member_page,
                headers={'referer': response.url})

        next_page = response.css('span.next').xpath('./a[@rel="next"]')
        next_page_url = next_page.xpath('@href').extract_first()

        if next_page_url is not None:
            yield scrapy.Request(
                urllib.parse.urljoin(response.url, next_page_url),
                callback=self.parse_page)

        for tag in self.__related_tags(response):
            yield scrapy.Request(
                '{}&type=illust'.format(
                    urllib.parse.urljoin(response.url, tag)),
                callback=self.parse_page)

    def parse_member_page(self, response):
        img = response.css('div.img-container').xpath('.//img')
        img_url = img.xpath('@src').extract_first()

        if img_url is not None:
            item = ImageScraperItem(
                file_urls=[img_url], files=[], response=response)

            yield item

    def __related_tags(self, response):
        related_tag = response.xpath('//ul[@class="tags"]/li/a[2]')
        tags = []

        for tag in related_tag:
            url = tag.xpath('@href').extract_first()
            parsed_url = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed_url.query)

            if qs['word'] in self.searched_tags:
                continue
            else:
                self.searched_tags.append(qs['word'])
                if url is not None:
                    tags.append(url)

        return tags
