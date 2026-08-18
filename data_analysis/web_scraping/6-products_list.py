#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import time
from selenium import webdriver


def scrape_products(url):
    """Launch a headless Chrome browser to scrape a static
    product page, using Selenium to extract titles, prices,
    descriptions, and ratings into a list of dictionaries."""
    # headless + fixed window size, exactly what the task asks for
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(1)

    products = []
    for card in driver.find_elements("class name", "thumbnail"):
        title = card.find_element("class name", "title")
        rating = card.find_element(
            "css selector", ".ratings p[data-rating]"
        )
        products.append({
            "title": title.get_attribute("title"),
            "price": card.find_element("class name", "price").text,
            "description": card.find_element(
                "class name", "description"
            ).text,
            "rating": int(rating.get_attribute("data-rating")),
        })

    return products
