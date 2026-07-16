#!/usr/bin/env python3
"""
Scrape the first page of quotes from quotes.toscrape.com
"""

from bs4 import BeautifulSoup
fetch_html = __import__('0-fetch_html').fetch_html


def scrape_basic(url):
    """
    Scrapes the first page of quotes.
    """
    html = fetch_html(url)

    soup = BeautifulSoup(html, "html.parser")

    quotes = []

    quote_blocks = soup.find_all("div", class_="quote")

    for quote in quote_blocks:
        text = quote.find("span", class_="text").get_text()

        author = quote.find("small", class_="author").get_text()

        tags = []
        tag_elements = quote.find_all("a", class_="tag")

        for tag in tag_elements:
            tags.append(tag.get_text())

        quotes.append({
            "text": text,
            "author": author,
            "tags": tags
        })

    return quotes
