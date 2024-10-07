# %% 
''' A hierarchical graph '''

# %%  special for the notebook - remove in the final version

# add to path the parent 
import sys
sys.path.append('../')

# load the environment variables
from dotenv import load_dotenv
load_dotenv()



# %% 

# from qs1 import AgentState1, GraphConfig, coder
# from qs1 import extract_code_from_output
from langgraph.graph import StateGraph, END

# %%
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated, Sequence
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model


class AgentState12_coder(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    codes: list[dict] = []

    code_imports: list[str] = [] # Store the imports from the code
    import_has_error: bool = False # if the code execution has an error

    code_to_execute: str = ""

    exec_has_error  : bool = False # if the code execution has an error
    exec_error_message : str = "" # error message if the code execution has an error

class AgentState12_CAb(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgentState12_CAa(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    list_agentState12_CAb: list[AgentState12_CAb] = []


class AgentState12(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    agentState12_CAa: AgentState12_CAa = None




from typing import TypedDict, Literal
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# %% node : coder_architect_A : gets user input and designs the NEXTJS14 architecture

def coder_architect_A(state: AgentState12_CAb, config: dict) -> AgentState12_CAb:
    # Get the messages from the state
    messages = state.messages
    system_prompt = "You are a helpful assistant that can write code."  # Define your system prompt

    # Prepend system prompt to messages
    messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Get the model name from config or default to "openai"
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)  # Replace with the appropriate model getter
    
    # Create a prompt template for the conversation
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{chat_history}\n\nHuman: {human_input}\n\nAssistant: Please provide your response, including any code if applicable.",
        input_variables=["system_prompt", "chat_history", "human_input"]
    )

    # Format chat history by extracting roles and content from the message sequence
    chat_history = ""
    for m in messages[1:-1]:
        # Check if the message is a dictionary with 'role' and 'content'
        if isinstance(m, dict):
            chat_history += f"{m['role']}: {m['content']}\n"
        # Check if it's an AIMessage or a different object and use 'content' only
        elif hasattr(m, 'content'):
            chat_history += f"{m.__class__.__name__}: {m.content}\n"
    
    # Access the last message content (human input)
    human_input = messages[-1]['content'] if isinstance(messages[-1], dict) else messages[-1].content

    # If the previous execution had an error, add a message to the chat history
    if state.exec_has_error:
        error_message = state.exec_error_message
        chat_history += f"\n\nPrevious execution had an error. Please fix the code and try again. Error message: {error_message}"
    
    # Format the final prompt
    formatted_prompt = prompt.format(
        system_prompt=system_prompt,
        chat_history=chat_history,
        human_input=human_input
    )
    
    # Invoke the model with the formatted prompt
    response = model.invoke(formatted_prompt)
    
    # Extract the content from the model response
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Append the assistant's response

# %% node : coder (superseding the one from qs1.py)
from langchain.prompts import PromptTemplate

import re

def extract_code_from_output(output: str) -> dict:
    """
    Extracts code, language, and generates a filename from the LLM output, 
    assuming code is inside triple backticks (e.g., ```python ... ```).
    
    Returns a dictionary with the structure: 
    {"language": <language>, "filename": <filename>, "code": <code>}.
    """
    print(output)
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
    
    print("No code found in the response")
    return None  # If no code is found, return None

import re

def extract_imports(code: str):
    # Regular expression for capturing import and from-import statements
    import_pattern = r'^\s*(import\s+\w+|from\s+\w+\s+import\s+\w+)'
    
    # Find all matches for import statements
    imports = re.findall(import_pattern, code, re.MULTILINE)
    
    return imports

# Example usage
# code_string = '''
# import os
# import sys
# from collections import defaultdict
# from math import sqrt
# def my_function():
#     pass
# '''

# imports = extract_imports(code_string)
# print(imports)
def coder(state: AgentState12_coder, config: dict) -> AgentState12_coder:
    # Get the messages from the state
    messages = state.messages
    system_prompt = "You are a helpful assistant that can write code."  # Define your system prompt

    # Prepend system prompt to messages
    messages = [{"role": "system", "content": system_prompt}] + messages
    
    # Get the model name from config or default to "openai"
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)  # Replace with the appropriate model getter
    
    # Create a prompt template for the conversation
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{chat_history}\n\nHuman: {human_input}\n\nAssistant: Please provide your response, including any code if applicable.",
        input_variables=["system_prompt", "chat_history", "human_input"]
    )

    # Format chat history by extracting roles and content from the message sequence
    chat_history = ""
    for m in messages[1:-1]:
        # Check if the message is a dictionary with 'role' and 'content'
        if isinstance(m, dict):
            chat_history += f"{m['role']}: {m['content']}\n"
        # Check if it's an AIMessage or a different object and use 'content' only
        elif hasattr(m, 'content'):
            chat_history += f"{m.__class__.__name__}: {m.content}\n"
    
    # Access the last message content (human input)
    human_input = messages[-1]['content'] if isinstance(messages[-1], dict) else messages[-1].content

    # If the previous execution had an error, add a message to the chat history
    if state.exec_has_error:
        error_message = state.exec_error_message
        chat_history += f"\n\nPrevious execution had an error. Please fix the code and try again. Error message: {error_message}"
    
    # Format the final prompt
    formatted_prompt = prompt.format(
        system_prompt=system_prompt,
        chat_history=chat_history,
        human_input=human_input
    )
    
    # Invoke the model with the formatted prompt
    response = model.invoke(formatted_prompt)
    
    # Extract the content from the model response
    response_content = response.content if hasattr(response, 'content') else str(response)
    
    # Append the assistant's response to the state
    state.messages.append({"role": "assistant", "content": response_content})
    
    # Extract the code from the response, if any
    code = extract_code_from_output(response_content)
    if code:
        # If code is found, append it to the state's codes list
        state.codes.append(code)
    else:
        print("No code found in the response")
    
    # Return the updated state
    return state



# %%
