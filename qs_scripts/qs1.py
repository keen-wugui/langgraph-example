# %%  special for the notebook - remove in the final version

# add to path the parent 
import sys
sys.path.append('../')

# load the environment variables
from dotenv import load_dotenv
load_dotenv()



# %% 
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
from my_agent.utils.state import AgentState

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# %% state 


class AgentState1(AgentState):
    codes: list[dict] = []


# %% output parser to parse code from the response
import re

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

# %%
system_prompt = """You are a helpful software assistant"""


from langchain.prompts import PromptTemplate


def coder(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)

    # Create a prompt template
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{chat_history}\n\nHuman: {human_input}\n\nAssistant: Please provide your response, including any code if applicable.",
        input_variables=["system_prompt", "chat_history", "human_input"]
    )

    # Format the chat history based on message type (check if it's an object or dict)
    chat_history = "\n".join([f"{m['role']}: {m['content']}" if isinstance(m, dict) else f"{m.role}: {m.content}" for m in messages[1:-1]])

    # Access the last message content correctly (handle object or dict)
    human_input = messages[-1]['content'] if isinstance(messages[-1], dict) else messages[-1].content

    # Format the prompt
    formatted_prompt = prompt.format(
        system_prompt=system_prompt,
        chat_history=chat_history,
        human_input=human_input
    )

    # Invoke the model
    response = model.invoke(formatted_prompt)

    # Extract the content from the response
    response_content = response.content if hasattr(response, 'content') else str(response)

    # Add the response to the state as a message
    state["messages"].append({"role": "assistant", "content": response_content})

    # Parse the code from the response and add it to the state
    code = extract_code_from_output(response_content)
    if code:
        state["codes"].append(code)
    else:
        print("No code found in the response")

    return {"messages": [{"role": "assistant", "content": response_content}]}

# example use 
# user_question = "Write a python code that generates a simply matplotlib plot with random data"
# state = AgentState1(messages=[{"role": "user", "content": user_question}], codes=[])
# config = {"configurable": {"model_name": "openai"}}
# coder(state, config)

# # %%
# state
# # %%
# state['messages']




# # %%
# state['codes'][0]['code']

# # %%
# exec(state['codes'][0]['code'])

# %%
