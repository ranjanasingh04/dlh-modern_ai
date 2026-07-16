#!/usr/bin/env python3
"""
Scrape quotes from all paginated pages of quotes.toscrape.com.
"""

from bs4 import BeautifulSoup
import time
from urllib import parse

fetch_html = __import__('0-fetch_html').fetch_html
scrape_basic = __import__('1-scrape_basic').scrape_basic


def scrape_paginated(base_url):
    """
    Scrape quotes from all pages, following each available Next link.
    """
    all_quotes = []
    current_url = base_url

    while current_url:
        page_quotes = scrape_basic(current_url)
        all_quotes.extend(page_quotes)

        html = fetch_html(current_url)
        soup = BeautifulSoup(html, 'html.parser')

        next_link = soup.select_one('li.next a')

        if next_link is None:
            current_url = None
        else:
            next_href = next_link.get('href')
            current_url = parse.urljoin(current_url, next_href)
            time.sleep(1)

    return all_quotes
