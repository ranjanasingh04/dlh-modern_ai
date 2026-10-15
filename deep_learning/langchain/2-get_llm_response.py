#!/usr/bin/env python3
"""Get a text response from a LangChain language model."""


def get_llm_response(llm, prompt):
    """Return the model's response to a prompt.

    Args:
        llm: An initialized LangChain language model.
        prompt: The input text to send to the model.

    Returns:
        The response as a string.
    """
    response = llm.invoke(prompt)

    # Chat models such as ChatMistralAI return an AIMessage.
    if hasattr(response, "content"):
        return response.content

    # HuggingFacePipeline returns a string.
    return response
