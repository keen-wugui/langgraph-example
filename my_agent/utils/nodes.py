from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from typing import TypedDict, List, Dict, Any
import re



system_prompt = """You are a helpful software assistant"""

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "anthropic":
        return ChatAnthropic(model="claude-3-opus-20240229")
    elif model_name == "openai":
        return ChatOpenAI(model="gpt-4-0125-preview")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def extract_code_from_output(output: str) -> dict:
    """
    Extracts code, language, and generates a filename from the LLM output, 
    assuming code is inside triple backticks (e.g., ```python ... ```).
    
    Returns a dictionary with the structure: 
    {"language": <language>, "filename": <filename>, "code": <code>}.
    """
    code_block = re.search(r"```(\w+)(.*?)```", output, re.DOTALL)
    if code_block:
        language = code_block.group(1).strip()  # Extract the language (e.g., python)
        code = code_block.group(2).strip()      # Extract the code inside the backticks
        
        # Generate a simple filename based on the language (this can be customized)
        # For example, we'll use "example" as the base name and append the appropriate file extension
        extension = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp"
        }.get(language, ".txt")  # Default to .txt if the language is unknown
        
        # You can customize how you generate filenames here, this is a simple example
        filename = f"example{extension}"
        
        return {"language": language, "filename": filename, "code": code}
    
    return None  # If no code is found, return None


# 


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model =  ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(tools)
    return model

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)