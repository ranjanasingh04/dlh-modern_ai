#!/usr/bin/env python3
"""Create a reusable LangChain prompt template."""

from langchain import prompts


def create_prompt_template(template_str, input_variables):
    """Return a prompt template with the specified input variables."""
    template = prompts.PromptTemplate(
        template=template_str,
        input_variables=input_variables,
    )
    return template
