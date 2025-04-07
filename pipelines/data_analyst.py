"""
title: Data Analyst
author: User
description: Data Analyst
required_open_webui_version: 0.4.3
version: 0.1
licence: MIT
"""

import os
from typing import List, Union, Generator, Iterator
from logging import getLogger
from langchain_core.messages import AIMessage
from pydantic import BaseModel
import sys

# Define the path to the external src directory
SRC_PATH = os.environ.get(
    "LANGGRAPH_SRC_PATH",  # Check for environment variable first
    "/Users/pisek/Documents/case-done-github/langgraph-agents/"  # Default path as fallback
)

# Verify the path exists before adding to sys.path
if os.path.exists(SRC_PATH):
    print(f"Using source directory: {SRC_PATH}")
    # Add the path to sys.path if not already there
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    
    try:
        from src.agents.data_analyst import get_llm, create_agent_builder, Valves
        from src.ui.openwebui import convert_messages_from_openwebui_to_langchain, convert_messages_from_langchain_to_openwebui
        print("Successfully imported modules from external src directory")
    except ImportError as e:
        print(f"IMPORT ERROR: {e}")
        print(f"Available modules in {SRC_PATH}: {os.listdir(SRC_PATH)}")
        if os.path.exists(os.path.join(SRC_PATH, 'src')):
            print(f"Contents of src directory: {os.listdir(os.path.join(SRC_PATH, 'src'))}")
        else:
            print(f"No 'src' directory found in {SRC_PATH}")
else:
    print(f"ERROR: Source path does not exist: {SRC_PATH}")
    print("Please set the LANGGRAPH_SRC_PATH environment variable to the correct path")
    raise ImportError(f"Source path not found: {SRC_PATH}")


logger = getLogger(__name__)
logger.setLevel("DEBUG")


class Pipeline:
    """
    Data Analyst pipeline.

    This pipeline is a simple LangGraph-based data analyst.

    Valve Parameters: NONE

    """

    class Valves(Valves):
        pass

    def __init__(self):
        self.name = "Data Analyst"

        # Initialize valve parameters
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        
        # Build the agent graph
        self.graph = create_agent_builder(
            llm=get_llm()
        ).compile()

    async def on_startup(self):
        logger.debug(f"on_startup:{self.name}")
        pass

    async def on_shutdown(self):
        logger.debug(f"on_shutdown:{self.name}")
        pass

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function. Processes user messages through the LangGraph agent.
        
        Handles conversion between OpenWebUI message format and LangGraph/LangChain message format.
        OpenWebUI format: List of dicts with 'role' and 'content' keys
        LangGraph/LangChain format: List of BaseMessage objects (SystemMessage, HumanMessage, AIMessage, ToolMessage)
        """
        # logger.debug(f"pipe:{self.name}")
        # logger.info(f"User Message: {user_message}")
        
        # Convert OpenWebUI messages to LangChain format
        from langchain_core.messages import convert_to_messages
        langchain_messages = convert_to_messages(messages)

        payload = {"messages": langchain_messages}
        try:
            results = self.graph.invoke(payload)
            new_messages = results["messages"][len(langchain_messages):]
        except Exception as e:
            msg = f"Error in pipe: {str(e)}"
            logger.error(msg, exc_info=True)
            new_messages = [AIMessage(content=msg)]

        # Convert LangChain messages back to OpenWebUI format
        # openwebui_messages = convert_messages_from_langchain_to_openwebui(new_messages)
        from langchain_core.messages import messages_to_dict
        openwebui_messages = messages_to_dict(new_messages)
        for msg in openwebui_messages:
            _data = msg["data"]
            for key, value in _data.items():
                msg[key] = value
        
        if body.get("stream", True):
            # logger.debug(f"LangGraph STREAM Response: BELOW")
            # Simulate streaming: yield each message content
            for msg in openwebui_messages:
                if "content" in msg and msg["content"]:
                    yield msg["content"]+"\n"
                if "function_call" in msg:
                    yield f"[Function Call: {msg['function_call']}]"+"\n"
        else:
            # Join all message contents into one response
            response = "\n\n".join(
                msg["content"] for msg in openwebui_messages if "content" in msg and msg["content"]
            )
            # logger.debug(f"LangGraph JOIN Response: {response}")
            return response