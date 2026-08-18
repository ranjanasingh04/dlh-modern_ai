#!/usr/bin/env python3
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

def scrape_products_list(url):
    """
    Scrapes a static product category page and returns a list of dictionaries.
    Each dictionary contains the title, price, description, and rating.
    """
    # Set up Chrome options for headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")

    # Initialize the driver
    driver = webdriver.Chrome(options=chrome_options)
    
    products = []

    try:
        driver.get(url)
        # Brief pause to ensure the page is fully loaded
        time.sleep(1)

        # Locate all product containers (the class used is 'thumbnail')
        product_elements = driver.find_elements(By.CLASS_NAME, "thumbnail")

        for element in product_elements:
            # 1. Title: title attribute of the <a> tag
            # Note: We find the <a> tag inside the h4 or directly in the caption
            title_link = element.find_element(By.TAG_NAME, "a")
            title = title_link.get_attribute("title")

            # 2. Price: text of <h4 class="price">
            price = element.find_element(By.CLASS_NAME, "price").text

            # 3. Description: text of <p class="description">
            description = element.find_element(By.CLASS_NAME, "description").text

            # 4. Rating: 'data-rating' attribute value of <p> under .ratings
            # CSS Selector selects a <p> that has the attribute data-rating inside .ratings
            rating_elem = element.find_element(By.CSS_SELECTOR, ".ratings p[data-rating]")
            rating_value = int(rating_elem.get_attribute("data-rating"))

            # Build the dictionary
            products.append({
                "title": title,
                "price": price,
                "description": description,
                "rating": rating_value
            })

    except Exception as e:
        # Proper error handling ensures driver quits even if a specific element fails
        raise e
    finally:
        driver.quit()

    return products
    