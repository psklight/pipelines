"""
title: LangGraph Chat Agent
author: User
description: Simple LangGraph-based chat agent
required_open_webui_version: 0.4.3
version: 0.1
requirements: langgraph, langchain-core, langchain-openai
licence: MIT
"""

from typing import List, Dict, Any, Annotated, TypedDict, Union, Generator, Iterator
from pydantic import BaseModel, Field
import os
from datetime import datetime
import time
from logging import getLogger

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

logger = getLogger(__name__)
logger.setLevel("DEBUG")

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next: str

class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
        MODEL_NAME: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
        SYSTEM_PROMPT: str = Field(
            default="You are a helpful assistant that provides concise and accurate information.",
            description="System prompt for the chat agent"
        )

    def __init__(self):
        self.name = "LangGraph Chat Agent"

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
        # Define the nodes
        def chat_node(state: AgentState) -> AgentState:
            """Process the chat messages and generate a response"""
            messages = state["messages"]
            # Get the LLM instance when needed
            response = self._get_llm().invoke(messages)
            return {"messages": messages + [response], "next": END}
        
        # Create the graph
        builder = StateGraph(AgentState)
        builder.add_node("chat", chat_node)
        
        # Define the edges
        builder.set_entry_point("chat")
        
        # Compile the graph
        return builder.compile()

    async def on_startup(self):
        logger.debug(f"on_startup:{self.name}")
        pass

    async def on_shutdown(self):
        logger.debug(f"on_shutdown:{self.name}")
        pass

    def _get_llm(self):
        """Get or create the LLM instance"""
        if not self.llm:
            # Initialize the LLM when needed
            self.llm = ChatOpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                model=self.valves.MODEL_NAME,
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
        
        # Convert messages to LangChain format
        langchain_messages = []
        
        # Add system message if not present
        if not any(msg.get("role") == "system" for msg in messages):
            langchain_messages.append(SystemMessage(content=self.valves.SYSTEM_PROMPT))
        
        # Add previous messages
        for msg in messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
        
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