#!/usr/bin/env python3
"""Initialize a Hugging Face text-to-text model using LangChain."""

from langchain_community import llms
from transformers import pipeline


def initialize_hf_llm(model_name, max_tokens):
    """Return a LangChain HuggingFacePipeline for text-to-text generation."""
    llm = llms.HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text2text-generation",
        model_kwargs={"dtype": "auto"},
        pipeline_kwargs={"max_new_tokens": max_tokens},
    )
    return llm
