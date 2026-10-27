#!/usr/bin/env python3
"""Create a FAISS vector store and retriever."""

from langchain_community import vectorstores


def create_faiss_vectorstore(
    docs, embedding_model, k, search_type="similarity"
):
    """Return a FAISS vector store and its configured retriever."""
    vectorstore = vectorstores.FAISS.from_documents(
        documents=docs,
        embedding=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )

    return vectorstore, retriever
