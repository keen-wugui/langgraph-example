# %% import shared 
from shared_states import * 
from shared_states import BaseState

# %% import necessary modules
import re
from dir_manager import DirManager, get_immediate_subfolders
from state_manager import StateManager
state_manager = StateManager()
from typing import Sequence, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from my_agent.utils.nodes import _get_model
from langchain_core.messages import ToolMessage

from s2_CAa_to_CAb import AgentState_CAb
import os
# %% graph state
# Define the state for the code generator interaction (Coder)
# AgentState_Coder class inheriting from BaseState
class AgentState_Coder(BaseState):
    assigned_func_des: str
    assigned_file: str  # From agentResponse_CAb.assigned_subtree_b
    assigned_file_path: str = ''  # From agentResponse_CAb.assigned_subtree_b
    assigned_subtree_c: dict = {}  # From agentResponse_CAb.assigned_subtree_b
    state_CAb: Optional[AgentState_CAb] = None
    existing_code: Optional[str] = None  # Existing code to be modified
    code_design_description: str  # Design description for code generation
    code: Optional[str] = Field(
        None,
        title="Code",
        description="The generated code. Do not include non-code content in this field."
    )

class CoderResponse(BaseModel):
    code_design_description: str  # The design description for which code needs to be generated
    code: Optional[str] = Field(None, title="Code", description="The generated code. Do not include non-code content in this field.")
    

tools = [CoderResponse]
# %% node
# Node: Generates code based on the design description
def coder(state: AgentState_Coder, config: dict) -> AgentState_Coder:
    # Extract messages from the current state
    messages = state.messages

    # Define the system prompt
    system_prompt = (
        "You are a code generator. Your job is to generate code based on the code design description provided. Provide your response using function: CoderResponse."
    )

    # Add the system prompt at the beginning of the conversation history
    messages = [{"role": "system", "content": system_prompt}] + messages

    # Choose the model based on the configuration or fall back to "openai"
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)  # Retrieve the model instance
    model = model.bind_tools(tools)  # Bind the tool node to the model

    # Create a prompt template that includes more contextual information
    prompt = PromptTemplate(
        template=(
            "{system_prompt}\n\n"
            "{chat_history}\n\n"
            "Context:\n"
            "Master Design Description:\n{master_design_description}\n\n"
            "Master Directory Tree:\n{master_dir_tree}\n\n"
            "Assigned Function Description:\n{assigned_func_des}\n\n"
            "Assigned Subtree:\n{assigned_subtree_c}\n\n"
            "Assigned File:\n{assigned_file}\n\n"
            "existing_code:\n{existing_code}\n\n"
            "---\n"
            "Human: {human_input}\n\n"
            "Assistant: Please provide the generated code."
        ),
        input_variables=[
            "system_prompt",
            "chat_history",
            "master_design_description",
            "master_dir_tree",
            "assigned_func_des",
            "assigned_file",
            "human_input",
            "existing_code",
            "assigned_subtree_c"
        ]
    )

    # Format the chat history from the message sequence, excluding system message and last message
    chat_history = ""
    for m in messages[1:-1]:  # Exclude the system prompt and the last human input
        if isinstance(m, dict):
            chat_history += f"{m['role'].capitalize()}: {m['content']}\n"
        elif hasattr(m, 'content'):
            chat_history += f"{m.role.capitalize()}: {m.content}\n"

    # Extract the human input from the last message in the sequence
    last_message = messages[-1]
    human_input = last_message['content'] if isinstance(last_message, dict) else last_message.content

    # Append any special note to human input
    if state.human_input_special_note:
        human_input += f"\n\nNote:\n{state.human_input_special_note}"

    # Extract contextual variables from the state
    master_design_description = state.state_CAb.master_design_description
    master_dir_tree = state.state_CAb.master_dir_tree
    assigned_func_des = state.assigned_func_des
    assigned_file = state.assigned_file

    # Invoke the model with the formatted prompt and get the response
    chain = prompt | model
    response = chain.invoke({
        'system_prompt': system_prompt,
        'chat_history': chat_history,
        'master_design_description': master_design_description,
        'master_dir_tree': master_dir_tree,
        'assigned_func_des': assigned_func_des,
        'assigned_file': assigned_file,
        'human_input': human_input,
        'existing_code': state.existing_code,
        'assigned_subtree_c': state.assigned_subtree_c
    })

    # Extract content from the model's response
    response_content = response.content if hasattr(response, 'content') else str(response)

    print('---response_content---')
    print(response_content)

    # Create the assistant's message
    assistant_message = BaseMessage(role="assistant", content=response_content, type="assistant")

    # Update the messages to include the assistant's response
    updated_messages = messages + [assistant_message]

    for tool_call in response.tool_calls:
        print('---tool_call---')
        print(tool_call)
        if tool_call["name"] == 'CoderResponse':
            this_coderResponse = CoderResponse(**tool_call["args"])
            print('CoderResponse successfully generated') 

            # overwrite the code 
            state.dir_manager.overwrite_file(directory=os.path.join(state.dir_manager.root_directory, state.assigned_file_path), 
                                                name="page.tsx", 
                                                content=this_coderResponse.code
                                                )
            # this_dir_manager = DirManager(project_params_path="project_params.json")

            # assigned_file_path = 'app/homepage'
            # this_dir_manager.overwrite_file(directory=os.path.join(this_dir_manager.root_directory, assigned_file_path), name="page.tsx", content="print('Hello, Universe!x')")


    # Update the code in the state
    return {
        'code_design_description':this_coderResponse.code_design_description,
        'code':this_coderResponse.code,
        'messages':[assistant_message],
        'count_invocations':state.count_invocations + 1,
    }
# %% retrying node
def retrying_node(state: AgentState_Coder) -> AgentState_Coder:
    state.count_invocations += 1
    print('---retry---')
    print('Incrementing count_invocations:', state.count_invocations)
    state.human_input_special_note += 'Please provide the code based on the code design description.'
    return state

# %% should_continue
def should_continue(state: AgentState_Coder) -> str:
    if state.count_invocations > 2:
        return 'end'
    if state.code is None:
        return 'retry'
    else:
        state.count_invocations += 1
        return 'end'

# %% compile subgraph
graph_builder = StateGraph(AgentState_Coder)

# Adding the node for coder process
graph_builder.add_node("coder", coder)

graph_builder.add_node("retrying_node", retrying_node)

# Setting the entry point for the graph
graph_builder.set_entry_point("coder")

# Setting the finish point for the graph
graph_builder.add_conditional_edges(
    "coder",
    should_continue,
    {
        "retry": "retrying_node",  # Retry in case of error
        "end": END ,   # Finish if no error
    },
)

graph_builder.add_edge("retrying_node", "coder")

# Compile the graph
graph = graph_builder.compile()

# %% test the graph


def extract_filename(description):
    # Regular expression to match the file name (ends with .js, .ts, .py, etc.)
    match = re.search(r'\b\w+\.\w{2,4}\b', description)
    
    if match:
        return match.group(0)
    else:
        return None

# Example usage:
# description = 'BlogList.js: Component responsible for displaying a list of blog posts.'
# filename = extract_filename(description)
# print(filename)  # Output: BlogList.js

# %% 

class UserMessage(BaseMessage):
    role: str
    content: str
    type: str = "user"

import os
if int(os.getenv('run_local_test', 1)):
    try: 
        state_s2
    except:
        from s2_CAa_to_CAb import new_state as state_s2

    import re

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
    new_state = graph.invoke(state, config)

    # %% 
    print(new_state['code'])

    # %%
    print(new_state['code_design_description'])

    # %%
    get_immediate_subfolders(state_s2['assigned_subtree_b'])

    # %% 

    sub_folder_keys
    # %% 
    get_immediate_subfolders(state_s2['assigned_subtree_b'])[sub_folder_keys[1]]
    # %%
    new_state['code']
    # %%
    new_state['dir_manager'].get_tree_structure()
# %%
