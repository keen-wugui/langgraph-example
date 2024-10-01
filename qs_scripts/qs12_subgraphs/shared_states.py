# %%  special for the notebook - remove in the final version

# add to path the parent 
import sys
sys.path.append('../')
sys.path.append('../../')

# load the environment variables
from dotenv import load_dotenv
load_dotenv()




# %%
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated, Sequence
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model


# class AgentState12_coder(BaseModel):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     codes: list[dict] = []

#     code_imports: list[str] = [] # Store the imports from the code
#     import_has_error: bool = False # if the code execution has an error

#     code_to_execute: str = ""

#     exec_has_error  : bool = False # if the code execution has an error
#     exec_error_message : str = "" # error message if the code execution has an error

# class AgentState12_CAb(BaseModel):
#     messages: Annotated[Sequence[BaseMessage], add_messages]




# class AgentState12(BaseModel):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
    
#     agentState12_CAa: AgentState12_CAa = None


# %%
