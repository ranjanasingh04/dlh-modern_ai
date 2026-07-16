#!/usr/bin/env python3
"""
Log in to a website and scrape authenticated quote data.
"""

import requests
from bs4 import BeautifulSoup


def login_and_scrape(login_url, user, pwd):
    """
    Log in and scrape quotes visible after authentication.

    Args:
        login_url (str): URL of the login page.
        user (str): Login username.
        pwd (str): Login password.

    Returns:
        list: A list of dictionaries containing quote information.
    """
    session = requests.Session()

    response = session.get(login_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    csrf_token = soup.find(
        "input",
        attrs={"name": "csrf_token"}
    )["value"]

    login_data = {
        "username": user,
        "password": pwd,
        "csrf_token": csrf_token
    }

    response = session.post(login_url, data=login_data)
    response.raise_for_status()

    response = session.get("https://quotes.toscrape.com/")
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = []

    for quote_block in soup.find_all("div", class_="quote"):
        text = quote_block.find("span", class_="text")
        author = quote_block.find("small", class_="author")
        tag_elements = quote_block.find_all("a", class_="tag")

        quotes.append({
            "text": text.get_text(strip=True),
            "author": author.get_text(strip=True),
            "tags": [
                tag.get_text(strip=True)
                for tag in tag_elements
            ]
        })

    return quotes
