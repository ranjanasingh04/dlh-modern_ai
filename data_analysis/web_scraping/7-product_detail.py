#!/usr/bin/env python3
"""This module provides utilities for web
scraping and collecting data from web resources."""
import time
from selenium import webdriver


def scrape_product_detail(url, delay=2.0):
    """Navigate to a single product's detail page
    using a headless browser, wait for the content to settle,
    and extract key details like title, price, description,
    and star rating using Selenium."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)
    driver.get(url)
    # wait for the page (stars are drawn by javascript)
    time.sleep(delay)

    # second h4 in .caption is the title
    title = driver.find_elements("css selector", ".caption h4")[1].text

    # first price h4 holds the price text
    price = driver.find_element("css selector", ".price").text

    # full product description
    description = driver.find_element(
        "css selector", ".description"
    ).text

    # each star is a <p>, count them for the rating
    stars = driver.find_elements(
        "css selector", ".ratings span.ws-icon-star"
    )
    rating = len(stars)

    return {
        "title": title,
        "price": price,
        "description": description,
        "rating": rating,
    }
