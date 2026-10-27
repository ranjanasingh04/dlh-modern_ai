#!/usr/bin/env python3
"""Module that adds session-aware chat history to a retrieval chain."""
from langchain_core import runnables


def create_session_aware_chat(retrieval_chain, get_session_history,
                              input_key, history_key, output_key):
    """Wrap a retrieval chain so each session keeps its own chat history."""
    session_chain = runnables.RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key=history_key,
        output_messages_key=output_key,
    )
    return session_chain
