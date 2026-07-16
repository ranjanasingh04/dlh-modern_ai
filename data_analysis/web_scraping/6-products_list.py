#!/usr/bin/env python3
"""
Scrape products from a static product page using Selenium.
"""

import time
from selenium import webdriver


def scrape_products_list(url):
    """
    Scrape products from a static category page.

    Args:
        url (str): Product category URL.

    Returns:
        list: List of product dictionaries.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    products = []

    try:
        driver.get(url)
        time.sleep(2)

        by = webdriver.common.by.By

        product_blocks = driver.find_elements(
            by.CSS_SELECTOR,
            ".thumbnail"
        )

        for product in product_blocks:
            title = product.find_element(
                by.CSS_SELECTOR,
                "a.title"
            )

            price = product.find_element(
                by.CSS_SELECTOR,
                "h4.price"
            )

            description = product.find_element(
                by.CSS_SELECTOR,
                "p.description"
            )

            rating = product.find_element(
                by.CSS_SELECTOR,
                ".ratings p[data-rating]"
            )

            products.append({
                "title": title.get_attribute("title"),
                "price": price.text,
                "description": description.text,
                "rating": int(rating.get_attribute("data-rating"))
            })

    finally:
        driver.quit()

    return products
