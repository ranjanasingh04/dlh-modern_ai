#!/usr/bin/env python3
"""
Fetch quote data from all API pages of quotes.toscrape.com.
"""

import json

fetch_html = __import__('0-fetch_html').fetch_html


def scrape_via_api(base_url):
    """
    Retrieve all quotes through the site's JSON API.
    """
    all_quotes = []
    page = 1
    has_next = True

    base_url = base_url.rstrip('/')

    while has_next:
        api_url = "{}/api/quotes?page={}".format(base_url, page)
        json_payload = fetch_html(api_url)
        data = json.loads(json_payload)

        for quote in data.get("quotes", []):
            all_quotes.append({
                "text": quote.get("text"),
                "author": quote.get("author", {}).get("name"),
                "tags": quote.get("tags", [])
            })

        has_next = data.get("has_next", False)
        page += 1

    return all_quotes
