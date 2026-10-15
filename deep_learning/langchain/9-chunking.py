#!/usr/bin/env python3
"""Split LangChain documents into smaller chunks."""

from langchain import text_splitter


def split_into_chunks(docs, chunk_size, chunk_overlap):
    """Return document chunks with the specified size and overlap."""
    splitter = text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
