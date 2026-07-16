#!/usr/bin/env python3
"""
Log in to quotes.toscrape.com and scrape quotes after authentication.
"""

import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """
    Log in to the website and scrape quotes from the protected page.

    Args:
        login_url (str): URL of the login page.
        user (str): Username used for authentication.
        pwd (str): Password used for authentication.

    Returns:
        list: Quote dictionaries containing text, author, and tags.
    """
    session = requests.Session()

    login_response = session.get(login_url)
    login_response.raise_for_status()

    soup = BeautifulSoup(login_response.text, "html.parser")
    csrf_input = soup.find("input", attrs={"name": "csrf_token"})

    if csrf_input is None or not csrf_input.get("value"):
        raise ValueError("CSRF token not found")

    csrf_token = csrf_input.get("value")

    credentials = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token
    }

    login_response = session.post(login_url, data=credentials)
    login_response.raise_for_status()

    protected_url = "https://quotes.toscrape.com/"
    quotes_response = session.get(protected_url)
    quotes_response.raise_for_status()

    soup = BeautifulSoup(quotes_response.text, "html.parser")
    quotes = []

    for quote_block in soup.find_all("div", class_="quote"):
        text_element = quote_block.find("span", class_="text")
        author_element = quote_block.find("small", class_="author")
        tag_elements = quote_block.find_all("a", class_="tag")

        quotes.append({
            "text": text_element.get_text(strip=True),
            "author": author_element.get_text(strip=True),
            "tags": [
                tag.get_text(strip=True)
                for tag in tag_elements
            ]
        })

    return quotes
