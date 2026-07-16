#!/usr/bin/env python3
"""
Extract quotes from JSON-LD blocks embedded in an HTML page.
"""

import json
from bs4 import BeautifulSoup

fetch_html = __import__('0-fetch_html').fetch_html


def extract_jsonld(url):
    """
    Extract quote data from JSON-LD script blocks.
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    quotes = []

    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        if not script.string:
            continue

        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue

        objects = data if isinstance(data, list) else [data]

        for obj in objects:
            if not isinstance(obj, dict):
                continue

            if obj.get("@type") != "Quote":
                continue

            keywords = obj.get("keywords", [])

            if isinstance(keywords, str):
                tags = [
                    keyword.strip()
                    for keyword in keywords.split(",")
                    if keyword.strip()
                ]
            elif isinstance(keywords, list):
                tags = keywords
            else:
                tags = []

            quotes.append({
                "text": obj.get("text"),
                "author": obj.get("author", {}).get("name"),
                "tags": tags
            })

    return quotes
