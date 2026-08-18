#!/usr/bin/env python3
import time
from selenium import webdriver

def scrape_products(url):
    """
    Scrapes a static product category page using only allowed imports.
    """
    # Initialize options directly from the webdriver module
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")

    # Initialize the driver
    driver = webdriver.Chrome(options=options)
    
    products = []

    try:
        driver.get(url)
        # Allow time for the static page to render
        time.sleep(1)

        # We use string literals for the locator strategy to avoid importing 'By'
        # mechanisms: "class name", "tag name", "css selector"
        product_elements = driver.find_elements("class name", "thumbnail")

        for element in product_elements:
            # 1. Title: from the 'title' attribute of the <a> tag
            title_link = element.find_element("tag name", "a")
            title = title_link.get_attribute("title")

            # 2. Price: text of <h4 class="price">
            price = element.find_element("class name", "price").text

            # 3. Description: text of <p class="description">
            description = element.find_element("class name", "description").text

            # 4. Rating: 'data-rating' attribute from the p tag inside .ratings
            rating_elem = element.find_element(
                "css selector", ".ratings p[data-rating]"
            )
            rating_value = int(rating_elem.get_attribute("data-rating"))

            # Append product dictionary to the list
            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating_value
            })

    except Exception as e:
        raise e
    finally:
        # Ensure the browser process is closed
        driver.quit()

    return products
