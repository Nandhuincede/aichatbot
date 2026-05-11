import os
from tools import ALL_TOOLS
from langchain_groq import ChatGroq


def _base_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env file")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=300
    )


def get_tool_calling_llm(tools=None):
    """LLM with bind_tools() — used in Step 2. The LLM decides which tool to call.
    Pass a specific tool list so the LLM only has access to the relevant tool.
    """
    bound_tools = tools if tools is not None else ALL_TOOLS
    return _base_llm().bind_tools(bound_tools)


def get_extraction_llm(schema):
    """LLM with structured output for a given extraction schema — used in Step 1."""
    return _base_llm().with_structured_output(schema)


def get_structured_llm(schema):
    """LLM with structured output for a given verdict schema — used in Step 3."""
    return _base_llm().with_structured_output(schema)

def get_reply_llm():
    """Plain LLM for generating conversational replies — used for all bot messages."""
    return _base_llm()