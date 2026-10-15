#!/usr/bin/env python3
"""Create a document answer chain with conversation history."""

from langchain_core import prompts
from langchain.chains import combine_documents


def document_answer_chain(llm, system_message, human_message):
    """Return a chat prompt and a document answer chain."""
    prompt = prompts.ChatPromptTemplate.from_messages([
        ("system", system_message),
        prompts.MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message),
    ])

    chain = combine_documents.create_stuff_documents_chain(llm, prompt)

    return prompt, chain
