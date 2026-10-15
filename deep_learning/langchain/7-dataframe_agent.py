#!/usr/bin/env python3
"""Create a LangChain Pandas DataFrame agent."""

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)


def create_dataframe_agent(llm, df):
    """Return an agent that can answer queries about a DataFrame."""
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        allow_dangerous_code=True,
        verbose=True,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
        },
    )
