#!/usr/bin/env python3
"""Create a reusable chat-style prompt template."""

from langchain import prompts


def create_chat_prompt_template(system_message, human_message):
    """Return a chat prompt containing system and human templates."""
    chat_template = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])
    return chat_template
