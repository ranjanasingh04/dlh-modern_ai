#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import time
from selenium import webdriver


def scroll_and_scrape(url, scroll_pause=2.0):
    """Open an infinite-scroll page in a headless browser,
    dynamically scroll to load all content, and extract unique
    product details using Selenium while filtering out duplicates."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # keep scrolling until the page height stops growing
    last_height = driver.execute_script(
        "return document.body.scrollHeight"
    )
    while True:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        # poll for the next batch instead of waiting the full
        # pause, so a fast page does not waste time each scroll
        waited = 0.0
        grew = False
        while waited < scroll_pause:
            time.sleep(0.2)
            waited += 0.2
            new_height = driver.execute_script(
                "return document.body.scrollHeight"
            )
            if new_height > last_height:
                last_height = new_height
                grew = True
                break
        # no new content within scroll_pause -> reached the end
        if not grew:
            break

    products = []
    seen = set()  # track (title, price) to skip duplicates
    for card in driver.find_elements("css selector", "div.thumbnail"):
        title = card.find_element(
            "css selector", "a.title"
        ).get_attribute("title")
        price = card.find_element("css selector", ".price").text

        # skip a product we already collected
        if (title, price) in seen:
            continue
        seen.add((title, price))

        description = card.find_element(
            "css selector", ".description"
        ).text
        rating = len(card.find_elements(
            "css selector", ".ratings span.ws-icon-star"
        ))

        products.append({
            "title": title,
            "price": price,
            "description": description,
            "rating": rating,
        })

    return products
