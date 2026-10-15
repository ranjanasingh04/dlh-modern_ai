#!/usr/bin/env python3
"""Build a sequence of system and human messages."""

from langchain_core import messages


def setup_message_sequence(system_msg, human_msgs):
    """Return a system message followed by human messages."""
    sysmessages = [messages.SystemMessage(content=system_msg)]
    for msg in human_msgs:
        sysmessages.extend([messages.HumanMessage(content=msg)])
    return sysmessages
