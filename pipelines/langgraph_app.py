"""
title: LangGraph App
author: User
description: Simple LangGraph-based chat agent
required_open_webui_version: 0.4.3
version: 0.1
licence: MIT
"""

from typing import List, Dict, Any, Annotated, TypedDict, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os
from logging import getLogger

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import custom implementation
import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom.langgraph_impl import build_graph, get_llm, convert_to_langchain_messages, OPENAI_API_KEY

logger = getLogger(__name__)
logger.setLevel("DEBUG")

class Pipeline:
    """
    LangGraph App pipeline.

    This pipeline is a simple LangGraph-based chat agent.

    Valve Parameters:

    * `OPENAI_API_KEY`: OpenAI API key (default: `OPENAI_API_KEY` environment variable)
    * `MODEL_NAME`: OpenAI model to use (default: "gpt-4o-mini")
    * `SYSTEM_PROMPT`: System prompt for the chat agent (default: "You are a helpful assistant that provides concise and accurate information.")

    """
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default=str(OPENAI_API_KEY), description="OpenAI API key")
        MODEL_NAME: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
        SYSTEM_PROMPT: str = Field(
            default="You are a helpful assistant that provides concise and accurate information.",
            description="System prompt for the chat agent"
        )

    def __init__(self):
        self.name = "LangGraph App"

        # Initialize valve parameters
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        
        # Initialize the LLM attribute to None - will be created when needed
        self.llm = None
        
        # Build the agent graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the chat agent"""
        return build_graph(self._get_llm())

    async def on_startup(self):
        logger.debug(f"on_startup:{self.name}")
        pass

    async def on_shutdown(self):
        logger.debug(f"on_shutdown:{self.name}")
        pass

    def _get_llm(self):
        """Get or create the LLM instance"""
        if not self.llm:
            # Initialize the LLM when needed using the custom implementation
            self.llm = get_llm(
                api_key=self.valves.OPENAI_API_KEY,
                model_name=self.valves.MODEL_NAME,
                temperature=0.7
            )
        return self.llm

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function. Processes user messages through the LangGraph agent.
        """
        logger.debug(f"pipe:{self.name}")
        logger.info(f"User Message: {user_message}")
        
        # Convert messages to LangChain format using the custom implementation
        langchain_messages = convert_to_langchain_messages(messages, self.valves.SYSTEM_PROMPT)
        
        # Initialize the state
        initial_state = {"messages": langchain_messages, "next": ""}
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract the assistant's response
        assistant_message = result["messages"][-1]
        
        # Return the response
        if body.get("stream", True):
            # Simulate streaming for now
            yield assistant_message.content
        else:
            return assistant_message.content