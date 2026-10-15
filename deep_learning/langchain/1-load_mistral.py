#!/usr/bin/env python3
"""Initialize a Mistral chat model using LangChain."""

import langchain_mistralai


def load_mistral(model_name, temperature):
    """Return a configured ChatMistralAI instance."""
    llm = langchain_mistralai.ChatMistralAI(
        model=model_name,
        temperature=temperature,
    )
    return llm
