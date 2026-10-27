#!/usr/bin/env python3
"""Create a history-aware retriever using LangChain."""

from langchain_core import prompts
from langchain import chains


def build_history_aware_retriever(llm, retriever, human_message):
    """Return a query prompt and a history-aware retriever."""
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", human_message),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        prompt,
    )

    return prompt, history_aware_retriever
