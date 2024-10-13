# %% 
import os
import jsonpickle
import uuid


class StateManager:
    def __init__(self, persist_dir='state_manager_states'):
        """
        Initializes the StateManager.
        :param persist_dir: Directory where states will be persisted.
        """
        self.persist_dir = persist_dir
        self._ensure_persist_dir_exists()

    def _ensure_persist_dir_exists(self):
        """Ensures the persistence directory exists."""
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

    def generate_state(self, subgraph_data):
        subgraph_id = str(uuid.uuid4())
        # Include 'id' directly in subgraph_data if it's a Pydantic model
        if hasattr(subgraph_data, 'id'):
            subgraph_data.id = subgraph_id
        else:
            subgraph_data['id'] = subgraph_id
        self.save_state(subgraph_id, subgraph_data)
        return subgraph_id

    def save_state(self, state_id, state):
        """
        Persists a given state to disk using jsonpickle.
        :param state_id: The unique ID of the state.
        :param state: The state data to be saved.
        """
        file_path = self._get_file_path(state_id)
        with open(file_path, 'w') as f:
            serialized_state = jsonpickle.encode(state)
            f.write(serialized_state)

    def load_state(self, state_id):
        """
        Loads a state from disk using jsonpickle.
        :param state_id: The unique ID of the state.
        :return: The state data.
        """
        file_path = self._get_file_path(state_id)
        with open(file_path, 'r') as f:
            serialized_state = f.read()
            state = jsonpickle.decode(serialized_state)
        return state

    def _get_file_path(self, state_id):
        """
        Generates the file path for a state based on its ID.
        :param state_id: The unique ID of the state.
        :return: The file path as a string.
        """
        return os.path.join(self.persist_dir, f'{state_id}.json')

    def restore_subgraph(self, state_id):
        state = self.load_state(state_id)
        # If state is a Pydantic model, access its attributes directly
        if hasattr(state, 'dict'):
            # It's a Pydantic model
            subgraph_data = state.dict()
        elif isinstance(state, dict):
            # It's a dictionary
            subgraph_data = state
        else:
            raise TypeError(f"Unexpected state type: {type(state)}")
        return subgraph_data

state_manager = StateManager()
# Example usage:
import os
if int(os.getenv('run_local_test', 0)):
    pass

    # Generate and save a new state
    subgraph_data = {
        'nodes': [1, 2, 3],
        'edges': [(1, 2), (2, 3)],
        'attributes': {'color': 'blue'}
    }
    state_id = state_manager.generate_state(subgraph_data)

    print(f"Generated state with ID: {state_id}")


    # from s1_human_to_CAa import new_state as state_s1
    # state_s1_id = state_manager.generate_state(state_s1)
    # print(f"Generated state with ID: {state_s1_id}") 


    # # Load and restore a subgraph from a state
    # restored_subgraph = state_manager.restore_subgraph(state_id)
    # print(f"Restored subgraph data: {restored_subgraph}")
# %%
