#!/usr/bin/env python3
"""Initialize a Hugging Face text-to-text model using LangChain."""

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


def initialize_hf_llm(model_name, max_tokens):
    """Return a LangChain HuggingFacePipeline for text-to-text generation."""
    hf_pipeline = pipeline(
        task="text2text-generation",
        model=model_name,
        max_new_tokens=max_tokens,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm
