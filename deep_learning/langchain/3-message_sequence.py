#!/usr/bin/env python3
"""Build a sequence of system and human messages."""

from langchain_core.messages import SystemMessage, HumanMessage


def setup_message_sequence(system_msg, human_msgs):
    """Return a system message followed by human messages."""
    messages = [SystemMessage(content=system_msg)]
    messages.extend(HumanMessage(content=msg) for msg in human_msgs)
    return messages
