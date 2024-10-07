# %% 
''' A hierarchical graph '''

# %% import shared 
from shared_states import * 

# %% 
from dir_manager import DirManager, dir_manager

# from qs1 import AgentState1, GraphConfig, coder
# from qs1 import extract_code_from_output
from langgraph.graph import StateGraph, END
import json
from typing import TypedDict, Literal
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]



# %% compile subgraph 1 for Code Architect A
from langgraph.graph import StateGraph, END
from typing import Sequence, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, validator, ValidationError, field_validator

from langchain.prompts import PromptTemplate
from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import ToolMessage

# Define the configuration options for the graph
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

# %% 

class AgentResponse_CAa(BaseModel):
    content: str
    folder_structure_dict: Union[str, dict] = Field(
        ...,
        title="Folder Structure",
        description="The folder structure designed by the assistant in nested dict/json format."
    )
    master_design_description: str = Field(
        ...,
        title="Master Design Description",
        description="Description of the design decisions made by the assistant."
    )

    @field_validator('folder_structure_dict', mode='before')
    def parse_folder_structure_dict(cls, v):
        if isinstance(v, str):
            try:
                # Attempt to parse the JSON string into a dictionary
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError('folder_structure_dict must be a valid JSON string') from e
        elif isinstance(v, dict):
            # If it's already a dictionary, return it as is
            return v
        else:
            raise TypeError('folder_structure_dict must be a dict or a JSON string representing a dict')

# passing the pydantic object as a tool for the model to use 
tools = [AgentResponse_CAa]


# %% graph state 
# Define the state for the code architect interaction (Architect A)
from typing import Optional, Sequence
class AgentState12_CAa(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    agentResponse_CAa:  Optional[AgentResponse_CAa] = None 
    content: str = None
    folder_structure_dict: Union[str, dict] = Field(
        default=None,
        title="Folder Structure",
        description="The folder structure designed by the assistant in nested dict/json format."
    )
    master_design_description: str = Field(
        default=None,
        title="Master Design Description",
        description="Description of the design decisions made by the assistant."
    )

    count_invocations: int = 0
    should_retry: bool = False
    human_input_special_note: str = ''

    dir_manager: DirManager = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
    

# %% node
# Subgraph node: Handles input from the user and designs the NEXTJS14 architecture
def coder_architect_A(state: AgentState12_CAa, config: dict) -> AgentState12_CAa:
    # Extract messages from the current state
    messages = state.messages
    system_prompt = "You are a helpful assistant that can write code."  # The system-level directive

    # Add the system prompt at the beginning of the conversation history
    messages = [{"role": "system", "content": system_prompt}] + messages

    # Choose the model based on the configuration or fall back to "openai"
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)  # Retrieve the model instance
    model_with_tool = model.bind_tools(tools)  # Bind the tool node to the model

    # Create a prompt template that structures the assistant's responses
    prompt = PromptTemplate(
        template="{system_prompt}\n\n{chat_history}\n\nHuman: {human_input}\n\nAssistant: Please provide your response, including any code if applicable. Respond using the tool: AgentResponse_CAa. Use the tool: UseDirManager to add, remove, or overwrite files in the directory.",
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
    if state.human_input_special_note != '':  # Check if there are any special notes to add
        human_input += '\n\n--\n\n' + state.human_input_special_note  # Add any special notes to the human input

    # Invoke the chosen model with the formatted prompt and get the response
    # chain =  prompt | model 
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

    # parse the response to get the multiTopics instance and add it to the state
    this_agentResponse_CAa = None
    for tool_call in response.tool_calls:
        print('---tool_call---')
        print(tool_call)
        print('\n---\n')

        if tool_call["name"] == 'AgentResponse_CAa':
            try:
                print('---AgentResponse_CAa tool called---')
                print(tool_call["args"])
                print('\n---\n')
                this_agentResponse_CAa = AgentResponse_CAa(**tool_call["args"])
                # comparing the new tree with the existing tree (get from folder)
                print('---comparing new tree---')
                state.dir_manager.compare_new_tree(this_agentResponse_CAa.folder_structure_dict)
                print('architect_response_tool successfully generated') 

            except ValidationError as e:
                print('---ValidationError---')
                print(e)
                state.human_input_special_note = '''
Please make sure the folder structure (folder_structure_dict) is in the correct format (nested dict/json) and the master design description is a string.
'''
                state.should_retry = True
                break 
            except Exception as e:
                print('---Exception---')
                print(e)
                state.human_input_special_note = f'''Please review error in the tool_call["args"]. Error: {e}'''
                state.should_retry = True
                break

            
        
        if tool_call["name"] == 'UseDirManager':
            print('UseDirManager tool called')

    # Return the updated state to include the new response and message history
    return {
        "messages": [assistant_message],
        "agentResponse_CAa": this_agentResponse_CAa,
        'dir_manager': state.dir_manager,
        'human_input_special_note': state.human_input_special_note,
        'should_retry': state.should_retry,

        'content': this_agentResponse_CAa.content,
        'folder_structure_dict': this_agentResponse_CAa.folder_structure_dict,
        'master_design_description': this_agentResponse_CAa.master_design_description,


    }

# %% retrying node 
def retrying_node(state: AgentState12_CAa) -> AgentState12_CAa:
    state.count_invocations += 1
    print('---retry---')
    print('Incrementing count_invocations:', state.count_invocations)
    # state.human_input_special_note = 'Please use tool: architect_response_tool to provide the folder structure and design description.'
    state.human_input_special_note += 'Please use tool: AgentResponse_CAa to provide the folder structure and (master) design description.'
    
    return {
        'count_invocations': state.count_invocations,
        'human_input_special_note': state.human_input_special_note,
        'should_retry': False,
    }

# %% should_continue
def should_continue(state: AgentState12_CAa) -> bool:
    if state.count_invocations > 2:
        return 'end'
    if state.should_retry:
        return 'retry'
    if state.agentResponse_CAa is None:
        return 'retry'

    else:
        state.count_invocations += 1
        return 'end'
    return 'end'

# %%  compile subgraph
graph_builder = StateGraph(AgentState12_CAa)

# Adding the node for architect's process
graph_builder.add_node("coder_architect_A", coder_architect_A)

graph_builder.add_node("retrying_node", retrying_node)


# Setting the entry point for the graph
graph_builder.set_entry_point("coder_architect_A")

# Setting the finish point for the graph
# graph_builder.add_edge("coder_architect_A", END)
graph_builder.add_conditional_edges(
    "coder_architect_A",
    should_continue,
    {
        "retry": "retrying_node",  # Retry in case of error
        "end": END ,   # Finish if no error
    },
)

graph_builder.add_edge("retrying_node", "coder_architect_A")

# Compile the graph
graph = graph_builder.compile()

# %% test the graph
class UserMessage(BaseMessage):
    role: str
    content: str
    type: str = "user"

# Example user message
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
new_state = graph.invoke(state, config)


# %%
new_state
# %%
new_state['agentResponse_CAa']
# %%
# new_state['agentResponse_CAa'].content
# %%
print(new_state['agentResponse_CAa'].folder_structure_dict)
