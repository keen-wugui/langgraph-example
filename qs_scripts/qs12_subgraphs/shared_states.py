# %%  special for the notebook - remove in the final version

# add to path the parent 
import sys
sys.path.append('../')
sys.path.append('../../')

# load the environment variables
from dotenv import load_dotenv
load_dotenv()

from dir_manager import DirManager, get_immediate_subfolders
from state_manager import StateManager


# %%
from pydantic import BaseModel, Field, ConfigDict
from typing import Sequence, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Annotated, Sequence
# from my_agent.utils.nodes import call_model, should_continue, tool_node, _get_model
import uuid 

# BaseState class with common attributes and StateManager integration
class BaseState(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[uuid.UUID] = None
    messages: Sequence[BaseMessage]
    count_invocations: int = 0
    human_input_special_note: str = ''
    dir_manager: Optional[DirManager] = None

    class Config:
        arbitrary_types_allowed = True

    def save_state(self, state_manager: StateManager):
        """
        Saves the current state using the StateManager.
        :param state_manager: An instance of StateManager.
        """
        state_manager.save_state(self.id, self)

    @classmethod
    def load_state(cls, state_id: str, state_manager: StateManager):
        """
        Loads a state using the StateManager.
        :param state_id: The unique ID of the state.
        :param state_manager: An instance of StateManager.
        :return: An instance of the state.
        """
        state = state_manager.load_state(state_id)
        return state
# %%
