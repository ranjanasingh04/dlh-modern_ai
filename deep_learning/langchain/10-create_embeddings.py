#!/usr/bin/env python3
"""Load a Hugging Face embedding model using LangChain."""

from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name):
    """Return a HuggingFaceEmbeddings instance for the specified model."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
