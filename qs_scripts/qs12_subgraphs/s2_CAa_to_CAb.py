# %% import shared 
from shared_states import * 

# %% import necessary modules
from dir_manager import DirManager, get_immediate_subfolders
from typing import Sequence, List, Literal, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage

# %% define the tool for AgentResponse_CAb
class AgentResponse_CAb(BaseModel):
    my_func_des: str = Field(..., title="Main Functional Description", description="The functional description of the assigned part of the code directory.")
    list_sub_func_des: List[str] = Field(..., title="List of Sub Functional Descriptions", description="List of sub-functional descriptions for subdirectories or code files in the assigned directory.")

# Passing the Pydantic object as a tool for the model to use
tools = [AgentResponse_CAb]

# %% graph state
# Define the state for the code architect interaction (Architect B)
class AgentState_CAb(BaseModel):
    master_design_description: str  # from agentResponse_CAa.design_description
    master_dir_tree: dict            # from agentResponse_CAa.folder_structure
    assigned_subtree_b: dict          # the tree structure part assigned to one CAb
    my_func_des: Optional[str] = None             # the functional description of the assigned part
    list_sub_func_des: List[str] = []    # list of sub-functional descriptions for subdirectories or code files in the assigned directory
    
    messages: Sequence[BaseMessage]
    count_invocations: int = 0
    human_input_special_note: str = ''
    dir_manager: DirManager = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

# %% node
# Subgraph node: Handles input from the user and generates the main functional description
def coder_architect_B(state: AgentState_CAb, config: dict) -> AgentState_CAb:
    # Extract messages from the current state
    messages = state.messages
    system_prompt = """
    You are a code architect that manages a sub-directory as part of a bigger code directory. 
    Your job is to break down a master design description into the main functional description (about the sub-directory you manage) 
    and a list of sub functional descriptions (for subdirectories or code files inside the sub-directory you manage).
    """  # The system-level directive

    # Add the system prompt at the beginning of the conversation history
    messages = [{"role": "system", "content": system_prompt}] + messages

    # Choose the model based on the configuration or fall back to "openai"
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)  # Retrieve the model instance
    model_with_tool = model.bind_tools(tools)  # Bind the tool node to the model

    # Create a prompt template that structures the assistant's responses
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{chat_history}\n\nHuman: {human_input}\n\nAssistant: Please provide your response, including any code if applicable. Respond using the tool: AgentResponse_CAb.",
        input_variables=["system_prompt", "chat_history", "human_input"]
    )

    # Format the chat history from the message sequence
    chat_history = ""
    for m in messages[1:-1]:  # Exclude the system message and last message
        if isinstance(m, dict):  # Check if message is a dictionary
            chat_history += f"{m['role']}: {m['content']}\n"
        elif hasattr(m, 'content'):  # Check for other message formats
            chat_history += f"{m.__class__.__name__}: {m.content}\n"

    # Extract the human input from the last message in the sequence
    human_input = messages[-1]['content'] if isinstance(messages[-1], dict) else messages[-1].content

    # Include master_design_description, master_dir_tree, assigned_subtree_b in the human input
    additional_context = f"""

Master Design Description:
{state.master_design_description}

Master Directory Tree:
{str(state.master_dir_tree)}

Assigned Directory Tree:
{str(state.assigned_subtree_b)}
"""

    if state.human_input_special_note != '':
        human_input += '\n\n--\n\n' + state.human_input_special_note
    human_input += '\n\n' + additional_context

    # Invoke the chosen model with the formatted prompt and get the response
    chain = prompt | model_with_tool
    response = chain.invoke({
        'system_prompt': system_prompt,
        'chat_history': chat_history,
        'human_input': human_input
    })
    
    # Extract content from the model's response
    response_content = response.content if hasattr(response, 'content') else str(response)

    print('---response_content---')
    print(response_content)

    assistant_message = BaseMessage(role="assistant", content=response_content, type="assistant")

    # Parse the response to get the AgentResponse_CAb instance and add it to the state
    this_agentResponse_CAb = None
    for tool_call in response.tool_calls:
        print('---tool_call---')
        print(tool_call)
        if tool_call["name"] == 'AgentResponse_CAb':
            this_agentResponse_CAb = AgentResponse_CAb(**tool_call["args"])
            print('AgentResponse_CAb successfully generated') 

    # Return the updated state to include the new response and message history
    return AgentState_CAb(
        master_design_description=state.master_design_description,
        master_dir_tree=state.master_dir_tree,
        assigned_subtree_b=state.assigned_subtree_b,
        my_func_des=this_agentResponse_CAb.my_func_des if this_agentResponse_CAb else None,
        list_sub_func_des=this_agentResponse_CAb.list_sub_func_des if this_agentResponse_CAb else [],
        messages=[assistant_message],
        count_invocations=state.count_invocations,
        human_input_special_note=state.human_input_special_note
    )

# %% retrying node
def retrying_node(state: AgentState_CAb) -> AgentState_CAb:
    state.count_invocations += 1
    print('---retry---')
    print('Incrementing count_invocations:', state.count_invocations)
    state.human_input_special_note += 'Please use tool: AgentResponse_CAb to provide the main functional description and list of sub-functional descriptions.'
    return state

# %% should_continue
def should_continue(state: AgentState_CAb) -> str:
    if state.count_invocations > 2:
        return 'end'
    if state.my_func_des is None:
        return 'retry'
    else:
        state.count_invocations += 1
        return 'end'

# %% compile subgraph
graph_builder = StateGraph(AgentState_CAb)

# Adding the node for architect's process
graph_builder.add_node("coder_architect_B", coder_architect_B)

graph_builder.add_node("retrying_node", retrying_node)

# Setting the entry point for the graph
graph_builder.set_entry_point("coder_architect_B")

# Setting the finish point for the graph
graph_builder.add_conditional_edges(
    "coder_architect_B",
    should_continue,
    {
        "retry": "retrying_node",  # Retry in case of error
        "end": END ,   # Finish if no error
    },
)

graph_builder.add_edge("retrying_node", "coder_architect_B")

# Compile the graph
graph = graph_builder.compile()

# %% test the graph
try: 
    state_s1
except:
    from s1_human_to_CAa import new_state as state_s1


class UserMessage(BaseMessage):
    role: str
    content: str
    type: str = "user"

# Example output from CAa
# master_design_description = "This is the design description of the Next.js project with authentication and a blog system."
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
new_state = graph.invoke(state, config)

# Output the results
print("---Final State---")
print("Main Functional Description:")
print(new_state['my_func_des'])

# %%
# Get the tree structure once to avoid redundant calls
tree_dict = state_s1['dir_manager'].get_tree_structure()
get_immediate_subfolders(tree_dict).keys()
# %% selected to go to the next sub graph s3
tree_dict['app']['homepage']
# %%
