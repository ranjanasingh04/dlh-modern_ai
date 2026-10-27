#!/usr/bin/env python3
"""Module that builds a conversational retrieval chain with LangChain."""
from langchain import chains


def create_conversational_retrieval_chain(history_aware_retriever,
                                          document_chain):
    """Build a retrieval chain that answers using retrieved documents."""
    retrieval_chain = chains.create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )
    return retrieval_chain
