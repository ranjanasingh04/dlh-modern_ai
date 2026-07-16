#!/usr/bin/env python3
"""
Fetch the HTML content of a web page.
"""

import requests


def fetch_html(url, headers=None, timeout=10):
    """
    Fetch a web page and return its HTML content.
    """
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
    )
    response.raise_for_status()

    return response.text
