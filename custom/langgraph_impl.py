"""
Implementation details for the LangGraph Chat Agent
"""

from typing import List, Union, TypedDict
from logging import getLogger

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
import os

try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

try:
    OPENAI_API_KEY = os.getenv("APP_OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("APP_OPENAI_API_KEY not found in environment variables")
except Exception as e:
    logger.error(f"Error loading APP_OPENAI_API_KEY: {e}")
    OPENAI_API_KEY = None

logger = getLogger(__name__)

# Define the state for our graph
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    next: str

def build_graph(llm) -> StateGraph:
    """Build the LangGraph for the chat agent"""
    
    # Define the nodes
    def chat_node(state: AgentState) -> AgentState:
        """Process the chat messages and generate a response"""
        messages = state["messages"]
        # Use the provided LLM instance
        response = llm.invoke(messages)
        return {"messages": messages + [response], "next": END}
    
    # Create the graph
    builder = StateGraph(AgentState)
    builder.add_node("chat", chat_node)
    
    # Define the edges
    builder.set_entry_point("chat")
    
    # Compile the graph
    return builder.compile()

def get_llm(api_key: str = OPENAI_API_KEY, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
    """Create an LLM instance"""
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )

def convert_to_langchain_messages(messages: List[dict], system_prompt: str) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
    """Convert messages to LangChain format"""
    langchain_messages = []
    
    # Add system message if not present
    if not any(msg.get("role") == "system" for msg in messages):
        langchain_messages.append(SystemMessage(content=system_prompt))
    
    # Add previous messages
    for msg in messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
        elif msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
    
    return langchain_messages
