# %%  
from shared_states import * 

# export environment variables
import os 
os.environ['run_local_test'] = '0'

# %% 
try: graph_s1
except:
    from s1_human_to_CAa import graph as graph_s1, AgentState12_CAa
try: graph_s2
except:
    from s2_CAa_to_CAb import graph as graph_s2, AgentState_CAb
try: graph_s3
except:
    from s3_CAb_to_coder import graph as graph_s3, AgentState_Coder, extract_filename

# %% 
import re
import json
from dir_manager import DirManager, dir_manager, get_immediate_subfolders
from langgraph.graph import StateGraph, END
import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from typing import Sequence, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, validator, ValidationError, field_validator

from langchain.prompts import PromptTemplate
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage


# %%
class UserMessage(BaseMessage):
    role: str
    content: str
    type: str = "user"

# %% s1
tree_dict_str = json.dumps(dir_manager.get_tree_structure(), indent=2)
user_question = f"""
Design a Next.js project with authentication and a blog system, with the provided directory: {tree_dict_str}.
Remove folder named 'hello' if exists. 
Add a page "homepage" in /app with the nextjs format.
"""
user_message = UserMessage(role="user", content=user_question)

# Initialize the state for Code Architect A with a sequence of messages
state = AgentState12_CAa(messages=[user_message], dir_manager=dir_manager)

# Define a configuration dictionary
config = {"configurable": {"model_name": "openai"}}

# Invoke the graph, running the process through Code Architect A
state_s1 = graph_s1.invoke(state, config)

# %% s2 

master_design_description = state_s1['agentResponse_CAa'].master_design_description

master_dir_tree = state_s1['dir_manager'].get_tree_structure()

# Assigned directory to CAb
# assigned_subtree_b = """
# │   │   ├── blog/
# │   │   │   ├── BlogList.js
# │   │   │   ├── BlogPost.js
# │   │   │   └── BlogEditor.js
# """
assigned_subtree_b = state_s1['dir_manager'].get_tree_structure()['app']

# Example user message
user_message = UserMessage(role="user", content="")

# Initialize the state for Code Architect B
state = AgentState_CAb(
    master_design_description=master_design_description,
    master_dir_tree=master_dir_tree,
    assigned_subtree_b=assigned_subtree_b,
    messages=[user_message],
    dir_manager=state_s1['dir_manager']
)

# Define a configuration dictionary
config = {"configurable": {"model_name": "openai"}}

# Invoke the graph, running the process through Code Architect B
state_s2 = graph_s2.invoke(state, config)

# %% s3 
# Initialize the state for Coder
user_message = UserMessage(role="user", 
                            content="Please generate the code for the assigned file."
            )

# Extract the assigned function description from AgentState_CAb
assigned_func_des = state_s2['list_sub_func_des'][0]

# Get the sub-folder keys from the assigned directory tree 
# temporary manually assigned <-- change later 
sub_folder_keys = list(get_immediate_subfolders(state_s2['assigned_subtree_b']))
assigned_subtree_c = get_immediate_subfolders(state_s2['assigned_subtree_b'])[sub_folder_keys[1]]

# temporary manually fixaed path <-- change later 
# ref: new_state['dir_manager'].root_directory = path to 'my-nextjs-app-5'
assigned_file_path = 'app/homepage' # <-- must not have '/' at the start

# Extract the assigned file name using the 'extract_filename' function
assigned_file = extract_filename(str(assigned_subtree_c))

# get existing code from the assigned file
existing_code = state_s2['dir_manager'].get_existing_code(directory=assigned_file_path, name=assigned_file)
# Initialize 'code_design_description' from 'state_s2.main_func_des'
code_design_description = state_s2['my_func_des']

state = AgentState_Coder(
    assigned_func_des=assigned_func_des,
    assigned_file_path=assigned_file_path,  
    assigned_file=assigned_file,
    assigned_subtree_c=assigned_subtree_c,
    code_design_description=code_design_description,
    messages=[user_message],
    count_invocations=0,
    human_input_special_note='',
    existing_code=existing_code,
    state_CAb=state_s2,
    dir_manager=state_s2['dir_manager']

)

# Define a configuration dictionary
config = {"configurable": {"model_name": "openai"}}

# Invoke the graph, running the process through Coder
state_s3 = graph_s3.invoke(state, config)

