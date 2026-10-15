#!/usr/bin/env python3
"""Load text documents from a folder using LangChain."""

from langchain_community import document_loaders


def load_documents(folder_path, file_pattern):
    """Return documents from files matching the specified glob pattern."""
    loader = document_loaders.DirectoryLoader(
        path=folder_path,
        glob=file_pattern,
        loader_cls=document_loaders.TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()
