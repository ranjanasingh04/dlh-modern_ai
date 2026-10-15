#!/usr/bin/env python3
"""Initialize a LangChain agent with multiple tools."""

from langchain import agents
from langchain_community.agent_toolkits import load_tools


def load_agent_with_toolkit(llm, toolkit_names, prompt):
    """Return an agent executor configured with the requested tools."""
    tools = load_tools.load_tools(toolkit_names, llm=llm)

    agent = agents.create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    executor = agents.AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    return executor
