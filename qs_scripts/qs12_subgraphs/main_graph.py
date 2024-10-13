# %%  
# from shared_states import * 
from shared_states import DirManager, get_immediate_subfolders, BaseState

# add to path the parent 
import sys
sys.path.append('../')
sys.path.append('../../')
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field, ConfigDict
from typing import Sequence, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated, Sequence
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
import uuid 

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
# from qs_scripts.qs12_subgraphs.dir_manager import DirManager, get_immediate_subfolders
from qs_scripts.qs12_subgraphs.state_manager import StateManager
from langgraph.graph import StateGraph, END
import json
from typing import TypedDict, Literal, Dict, Any, Optional
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

state_manager = StateManager()
dir_manager = DirManager(project_params_path="project_params.json")
# %% s1
# tree_dict_str = json.dumps(dir_manager.get_tree_structure(), indent=2)
# user_question = f"""
# Design a Next.js project with authentication and a blog system, with the provided directory: {tree_dict_str}.
# Remove folder named 'hello' if exists. 
# Add a page "homepage" in /app with the nextjs format.
# """
# user_message = UserMessage(role="user", content=user_question)

# # Initialize the state for Code Architect A with a sequence of messages
# state = AgentState12_CAa(messages=[user_message], dir_manager=dir_manager)

# # Define a configuration dictionary
# config = {"configurable": {"model_name": "openai"}}

# # Invoke the graph, running the process through Code Architect A
# state_s1 = graph_s1.invoke(state, config)
# state_s1 = AgentState12_CAa(**state_s1)
# state_s1.dir_manager = dir_manager
# state_s1.save_state(state_manager)
# print('state_s1.id is ',state_s1.id)

# # %% s2 

# master_design_description = state_s1.agentResponse_CAa.master_design_description

# master_dir_tree = state_s1.dir_manager.get_tree_structure()

# # Assigned directory to CAb
# # assigned_subtree_b = """
# # │   │   ├── blog/
# # │   │   │   ├── BlogList.js
# # │   │   │   ├── BlogPost.js
# # │   │   │   └── BlogEditor.js
# # """
# assigned_subtree_b = state_s1.dir_manager.get_tree_structure()['app']

# # Example user message
# user_message = UserMessage(role="user", content="")

# # Initialize the state for Code Architect B
# state = AgentState_CAb(
#     master_design_description=master_design_description,
#     master_dir_tree=master_dir_tree,
#     assigned_subtree_b=assigned_subtree_b,
#     messages=[user_message],
#     dir_manager=state_s1.dir_manager
# )

# # Define a configuration dictionary
# config = {"configurable": {"model_name": "openai"}}

# # Invoke the graph, running the process through Code Architect B
# state_s2 = graph_s2.invoke(state, config)
# state_s2 = AgentState_CAb(**state_s2)
# state_s2.dir_manager = dir_manager
# state_s2.save_state(state_manager)
# print('state_s2.id is ',state_s2.id)
# # %% s3 
# # Initialize the state for Coder
# user_message = UserMessage(role="user", 
#                             content="Please generate the code for the assigned file."
#             )

# # temporary manually fixaed path <-- change later 

# sub_folder_keys = list(get_immediate_subfolders(state_s2.assigned_subtree_b))
# assigned_subtree_c = get_immediate_subfolders(state_s2.assigned_subtree_b)[sub_folder_keys[1]]

# assigned_func_des = state_s2.list_sub_func_des[0]
# assigned_file_path = 'app/homepage' # <-- must not have '/' at the start
# assigned_file = extract_filename(str(assigned_subtree_c))

# code_design_description = state_s2.my_func_des
# existing_code = state_s2.dir_manager.get_existing_code(directory=assigned_file_path, name=assigned_file)


# state = AgentState_Coder(
#     assigned_func_des=assigned_func_des,
#     assigned_file_path=assigned_file_path,  
#     assigned_file=assigned_file,
#     assigned_subtree_c=assigned_subtree_c,
#     code_design_description=code_design_description,
#     messages=[user_message],
#     count_invocations=0,
#     human_input_special_note='',
#     existing_code=existing_code,
#     state_CAb=state_s2,
#     dir_manager=state_s2.dir_manager

# )

# # Define a configuration dictionary
# config = {"configurable": {"model_name": "openai"}}

# # Invoke the graph, running the process through Coder
# state_s3 = graph_s3.invoke(state, config)
# state_s3 = AgentState_Coder(**state_s3)
# state_s3.dir_manager = dir_manager
# state_s3.save_state(state_manager)
# print('state_s3.id is ',state_s3.id)

# %% start of main graph 

# state 
class MainState(BaseState):
    messages: Optional[Sequence[BaseMessage]] = None
    # Register to save IDs of sub-states
    sub_state_ids: Dict[str, str] = Field(default_factory=dict)
    # Additional fields
    initial_user_message: Optional[Union[BaseMessage, UserMessage]] = None

# def invoke_s1(state: MainState, config: dict) -> MainState:
#     # Prepare the initial state for s1
#     user_message = state.initial_user_message
#     s1_state = AgentState12_CAa(messages=[user_message], dir_manager=state.dir_manager)
    
#     # Invoke subgraph s1
#     state_s1 = graph_s1.invoke(s1_state, config)
#     state_s1 = AgentState12_CAa(**state_s1)
#     state_s1.save_state(state_manager)
#     return {'state_s1':state_s1}

def invoke_s1(state: MainState, config: dict) -> Dict[str, Any]:
    # Prepare the initial state for s1
    user_message = state.initial_user_message
    s1_state = AgentState12_CAa(messages=[user_message], dir_manager=state.dir_manager)
    
    # Invoke subgraph s1
    state_s1 = graph_s1.invoke(s1_state, config)
    state_s1 = AgentState12_CAa(**state_s1)
    state_s1.save_state(state_manager)
    
    # Get the ID of state_s1 and update sub_state_ids
    state_s1_id = state_s1.id
    sub_state_ids = state.sub_state_ids.copy()
    sub_state_ids['state_s1'] = state_s1_id
    
    # Return the updated sub_state_ids to update MainState
    return {'sub_state_ids': sub_state_ids}

def should_continue_s1(state: MainState) -> str:
    if True:
        return 'invoke_s2'
    else:
        return END

# def invoke_s2(state: MainState, config: dict) -> MainState:
#     # Prepare the state for s2 using output from s1
#     s1_output = state.state_s1
#     s2_state = AgentState_CAb(
#         master_design_description=s1_output.master_design_description,
#         master_dir_tree=s1_output.dir_manager.get_tree_structure(),
#         assigned_subtree_b=s1_output.folder_structure_dict['app'],
#         messages=[],
#         dir_manager=s1_output.dir_manager
#     )
#     # Invoke subgraph s2
#     state_s2 = graph_s2.invoke(s2_state, config)
#     state_s2 = AgentState_CAb(**state_s2)
#     state_s2.save_state(state_manager)
#     return {'state_s2':state_s2}

def invoke_s2(state: MainState, config: dict) -> Dict[str, Any]:
    # Restore state_s1 using its ID
    state_s1_id = state.sub_state_ids['state_s1']
    if not state_s1_id:
        raise ValueError("state_s1_id not found in sub_state_ids")
    s1_state_data = state_manager.restore_subgraph(state_s1_id)
    s1_output = AgentState12_CAa(**s1_state_data)
    
    # Prepare the state for s2 using output from s1
    s2_state = AgentState_CAb(
        master_design_description=s1_output.master_design_description,
        master_dir_tree=s1_output.dir_manager.get_tree_structure(),
        assigned_subtree_b=s1_output.folder_structure_dict['app'],
        messages=[],
        dir_manager=s1_output.dir_manager,
        parent_id=state_s1_id # add parent_id to track the parent state
    )
    # Invoke subgraph s2
    state_s2 = graph_s2.invoke(s2_state, config)
    state_s2 = AgentState_CAb(**state_s2)
    state_s2.save_state(state_manager)
    
    # Get the ID of state_s2 and update sub_state_ids
    state_s2_id = state_s2.id
    sub_state_ids = state.sub_state_ids.copy()
    sub_state_ids['state_s2'] = state_s2_id
    
    return {'sub_state_ids': sub_state_ids}

def should_continue_s2(state: MainState) -> str:
    if True:
        return 'invoke_s3'
    else:
        return END

# def invoke_s3(state: MainState, config: dict) -> MainState:
#     # Prepare the state for s3 using output from s2
#     s2_output = state.state_s2
#     assigned_func_des = s2_output.list_sub_func_des[0]
#     assigned_file_path = 'app/homepage'
#     sub_folder_keys = list(get_immediate_subfolders(s2_output.assigned_subtree_b))
#     assigned_subtree_c = get_immediate_subfolders(s2_output.assigned_subtree_b)[sub_folder_keys[1]]
#     assigned_file = extract_filename(str(assigned_subtree_c))
#     existing_code = s2_output.dir_manager.get_existing_code(directory=assigned_file_path, name=assigned_file)
#     s3_state = AgentState_Coder(
#         assigned_func_des=assigned_func_des,
#         assigned_file_path=assigned_file_path,
#         assigned_file=assigned_file,
#         assigned_subtree_c=assigned_subtree_c,
#         code_design_description=s2_output.my_func_des,
#         messages=[],
#         existing_code=existing_code,
#         state_CAb=s2_output,
#         dir_manager=s2_output.dir_manager
#     )
#     # Invoke subgraph s3
#     state_s3 = graph_s3.invoke(s3_state, config)
#     state_s3 = AgentState_Coder(**state_s3)
#     state_s3.save_state(state_manager)
#     return {'state_s3':state_s3}

def invoke_s3(state: MainState, config: dict) -> Dict[str, Any]:
    # Restore state_s2 using its ID
    state_s2_id = state.sub_state_ids['state_s2']
    if not state_s2_id:
        raise ValueError("state_s2_id not found in sub_state_ids")
    s2_state_data = state_manager.restore_subgraph(state_s2_id)
    s2_output = AgentState_CAb(**s2_state_data)
    
    # Prepare the state for s3 using output from s2
    assigned_func_des = s2_output.list_sub_func_des[0]
    assigned_file_path = 'app/homepage'
    sub_folder_keys = list(get_immediate_subfolders(s2_output.assigned_subtree_b))
    assigned_subtree_c = get_immediate_subfolders(s2_output.assigned_subtree_b)[sub_folder_keys[1]]
    assigned_file = extract_filename(str(assigned_subtree_c))
    existing_code = s2_output.dir_manager.get_existing_code(directory=assigned_file_path, name=assigned_file)
    s3_state = AgentState_Coder(
        assigned_func_des=assigned_func_des,
        assigned_file_path=assigned_file_path,
        assigned_file=assigned_file,
        assigned_subtree_c=assigned_subtree_c,
        code_design_description=s2_output.my_func_des,
        messages=[],
        existing_code=existing_code,
        state_CAb=s2_output,
        dir_manager=s2_output.dir_manager,
        parent_id=state_s2_id   # Add parent_id to track the parent state
    )
    # Invoke subgraph s3
    state_s3 = graph_s3.invoke(s3_state, config)
    state_s3 = AgentState_Coder(**state_s3)
    state_s3.save_state(state_manager)
    
    # Get the ID of state_s3 and update sub_state_ids
    state_s3_id = state_s3.id
    sub_state_ids = state.sub_state_ids.copy()
    sub_state_ids['state_s3'] = state_s3_id
    
    return {'sub_state_ids': sub_state_ids}

# compile
main_graph_builder = StateGraph(MainState)

main_graph_builder.add_node('invoke_s1', invoke_s1)
main_graph_builder.add_node('invoke_s2', invoke_s2)
main_graph_builder.add_node('invoke_s3', invoke_s3)

main_graph_builder.set_entry_point('invoke_s1')

# Add conditional edges
main_graph_builder.add_conditional_edges(
    'invoke_s1',
    should_continue_s1,
    {
        'invoke_s2': 'invoke_s2',
        END: END,
    }
)

main_graph_builder.add_conditional_edges(
    'invoke_s2',
    should_continue_s2,
    {
        'invoke_s3': 'invoke_s3',
        END: END,
    }
)

main_graph_builder.add_edge('invoke_s3', END)

# Compile the graph
main_graph = main_graph_builder.compile()

# %% 
# Prepare the initial user message
user_question = """
Design a Next.js project with authentication and a blog system, with the provided directory.
Remove folder named 'hello' if exists.
Add a page "homepage" in /app with the nextjs format.
"""
initial_user_message = UserMessage(role="user", content=user_question)

# Initialize the main state
main_state = MainState(
    initial_user_message=initial_user_message,
    dir_manager=dir_manager
)

# Configuration dictionary
config = {"configurable": {"model_name": "openai"}}

# Invoke the main graph
final_state = main_graph.invoke(main_state, config)
# %%
final_state

# %%
state_manager.restore_subgraph(final_state['sub_state_ids']['state_s2'])
# %%
