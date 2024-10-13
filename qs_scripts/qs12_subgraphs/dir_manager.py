import os
import re
import shutil
import json
from deepdiff import DeepDiff


def get_immediate_subfolders(tree_dict):
    """
    Returns a dictionary of immediate subfolders and their contents from the given directory tree dictionary.

    Parameters:
    - tree_dict (dict): The directory tree structure returned by dir_manager.get_tree_structure().

    Returns:
    - subfolders_dict (dict): A dictionary where keys are subfolder names and values are their respective dictionaries.
    """
    # Create a dictionary of immediate subfolders and their contents
    subfolders_dict = {key: value for key, value in tree_dict.items() if isinstance(value, dict)}
    return subfolders_dict


class DirManager:
    def __init__(self, project_params_path, exclusions=None):
        # Load project parameters from the given JSON file
        with open(project_params_path, "r") as f:
            project_params = json.load(f)
        self.project_name = project_params['nextjs']['project_name']
        
        # Set the root directory to the project name
        self.root_directory = os.path.abspath(self.project_name)
        self.exclusions = exclusions if exclusions else ['node_modules', '__pycache__']
        
        # Initialize the tree dictionary with the project name
        self.tree_dict = {self.project_name: {}}

    def parse_tree_to_dict(self, tree):
        pattern = re.compile(r'^(?P<bars>(\|   )*)(\|-- )?(?P<name>.*)')
        lines = tree.strip().split("\n")
        root = {}
        stack = [root]
        levels = [-1]

        for line in lines:
            match = pattern.match(line)
            if match:
                bars = match.group('bars') or ''
                name = match.group('name').strip()
                level = bars.count('|   ')
                if match.group(3):
                    level += 1
                while level <= levels[-1]:
                    stack.pop()
                    levels.pop()
                current_level = stack[-1]
                if name.startswith("/"):
                    current_level[name] = {}
                    stack.append(current_level[name])
                    levels.append(level)
                else:
                    current_level[name] = ""
            else:
                name = line.strip()
                root[name] = {}
                stack = [root[name]]
                levels = [0]
        return root

    def parse_dict_to_tree(self, d):
        def helper(current_dict, prefix):
            lines = []
            keys = list(current_dict.keys())
            for idx, key in enumerate(keys):
                is_last = idx == len(keys) - 1
                connector = '|-- '
                line = prefix + connector + key
                lines.append(line)
                if isinstance(current_dict[key], dict) and current_dict[key]:
                    extension = '|   ' if not is_last else '    '
                    lines.extend(helper(current_dict[key], prefix + extension))
            return lines

        lines = []
        keys = list(d.keys())
        for idx, key in enumerate(keys):
            lines.append(key)
            if isinstance(d[key], dict) and d[key]:
                lines.extend(helper(d[key], ''))
        return '\n'.join(lines)

    def folder_tree_dict(self, directory=None):
        directory = directory or self.root_directory
        tree_dict = {}
        items = [item for item in os.listdir(directory) if not item.startswith(".")]
        items = [item for item in items if item not in self.exclusions]
        items.sort()
        for item in items:
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                tree_dict[item] = self.folder_tree_dict(path)
            else:
                tree_dict[item] = None
        return tree_dict

    def add_to_directory(self, directory, name, content=None, is_folder=False):
        # Remove leading slash from name
        try:
            name = name.lstrip('/')
        except:
            name = name
        full_path = os.path.join(directory, name)
        if is_folder:
            os.makedirs(full_path, exist_ok=True)
            print(f"Folder created: {full_path}")
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content or "")
            print(f"File created: {full_path}")

    def overwrite_file(self, directory, name, content):
        # Remove leading slash from name
        name = name.lstrip('/')
        file_path = os.path.join(directory, name)
        
        # Debug statements
        print(f"directory: {directory}")
        print(f"name: {name}")
        print(f"file_path: {file_path}")
        print(f"os.path.dirname(file_path): {os.path.dirname(file_path)}")
        
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the content to the file
        with open(file_path, "w") as f:
            f.write(content)
        print(f"File overwritten: {file_path}")

    def delete_item(self, directory, name, is_folder=False):
        # Remove leading slash from name
        name = name.lstrip('/')
        item_path = os.path.join(directory, name)
        if is_folder:
            if os.path.exists(item_path) and os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Folder deleted: {item_path}")
            else:
                print(f"Folder {item_path} does not exist.")
        else:
            if os.path.exists(item_path) and os.path.isfile(item_path):
                os.remove(item_path)
                print(f"File deleted: {item_path}")
            else:
                print(f"File {item_path} does not exist.")

    def get_tree_structure(self, return_str=False):
        tree_dict = self.folder_tree_dict()
        if return_str:
            return self.parse_dict_to_tree(tree_dict)
        else:
            return tree_dict
    
    def compare_new_tree(self, new_input):
        # Check if new_input is a dict or str
        print('---new_input---')
        print('type:', type(new_input))
        print(new_input)
        if isinstance(new_input, dict):
            new_tree_dict = new_input
        elif isinstance(new_input, str):
            new_tree_dict = self.parse_tree_to_dict(new_input)
        else:
            raise TypeError("new_input must be of type dict or str, instead it is of type: {}".format(type(new_input)))
        
        # Get the existing tree dictionary
        existing_tree_dict = self.get_tree_structure()
        
        # Compute the differences using DeepDiff
        diff = DeepDiff(existing_tree_dict, new_tree_dict, ignore_order=True)
        
        # Process and print the differences
        print("Differences between existing and new directory structures:")
        print(diff)
        print("---\n")
        
        if 'dictionary_item_added' in diff:
            print("\nItems added:")
            for item in diff['dictionary_item_added']:
                # Format the path for readability
                path = self.format_deepdiff_path(item)
                print(f"Added: {path}")
                
                # Determine if the added item is a folder or file
                is_folder = self.is_folder_in_tree(new_tree_dict, item)
                
                # Add the new item to the directory
                self.add_to_directory(directory=self.root_directory, name=path, is_folder=is_folder)

        if 'dictionary_item_removed' in diff:
            print("\nItems removed:")
            for item in diff['dictionary_item_removed']:
                path = self.format_deepdiff_path(item)
                print(f"Removed: {path}")
                
                # Determine if the removed item is a folder or file
                is_folder = self.is_folder_in_tree(existing_tree_dict, item)
                
                # Delete the removed item from the directory
                self.delete_item(directory=self.root_directory, name=path, is_folder=is_folder)

        if 'values_changed' in diff:
            print("\nValues changed:")
            for item in diff['values_changed']:
                change = diff['values_changed'][item]
                path = self.format_deepdiff_path(item)
                print(f"Warning: Changed: {path} - from {change['old_value']} to {change['new_value']}.")

        if not diff:
            print("No differences found between the existing and new directory structures.")

        # Update the stored tree dictionary
        self.tree_dict = new_tree_dict
        return new_tree_dict

    def is_folder_in_tree(self, tree_dict, deepdiff_path):
        # Convert DeepDiff path to a list of keys
        key_list = self.deepdiff_path_to_keys(deepdiff_path)
        
        # Navigate the tree_dict using the keys
        value = tree_dict
        try:
            for key in key_list:
                value = value[key]
        except KeyError:
            # Handle the error, possibly assume it's a file
            return False
        return isinstance(value, dict)

    def deepdiff_path_to_keys(self, deepdiff_path):
        # Remove 'root' and extract keys inside square brackets
        key_pattern = re.compile(r"\['(.*?)'\]")
        keys = key_pattern.findall(deepdiff_path)
        return keys

    def format_deepdiff_path(self, deepdiff_path):
        # Converts DeepDiff path to a file system path without leading slash
        path = deepdiff_path.replace("root", "").replace("['", "/").replace("']", "")
        # Remove leading slash if present
        path = path.lstrip('/')
        return path
    
    def get_existing_code(self, directory, name):
        """
        Reads the content of a file and returns it as a string.

        Parameters:
        - directory (str): The directory where the file is located.
        - name (str): The name of the file, including any subdirectories if necessary.

        Returns:
        - content (str): The content of the file.
        """
        # Remove leading slash from name
        name = name.lstrip('/')
        file_path = os.path.join(directory, name)

        # Debug statements
        print(f"directory: {directory}")
        print(f"name: {name}")
        print(f"file_path: {file_path}")

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None

        # Read the content of the file
        with open(file_path, "r") as f:
            content = f.read()
        print(f"File read: {file_path}")
        return content
# Assuming 'project_params.json' is in the current directory and contains the necessary data

# Initialize DirManager with the path to 'project_params.json'
dir_manager = DirManager(project_params_path="project_params.json")

# Print the project name
# print(f"Project Name: {dir_manager.project_name}")

# Get the current tree structure
# print(dir_manager.get_tree_structure())

# Add a new folder inside the project directory
# dir_manager.add_to_directory(directory=dir_manager.root_directory, name="hello", is_folder=True)

# Add a new file inside the new folder
# dir_manager.add_to_directory(directory=os.path.join(dir_manager.root_directory, "hello"), name="hello.py", content="print('Hello, World!')")

# Overwrite the existing file
# assigned_file_path = 'app/homepage'
# dir_manager.overwrite_file(directory=os.path.join(dir_manager.root_directory, assigned_file_path), name="page.tsx", content="print('Hello, Universe!x')")

# Delete the file
# dir_manager.delete_item(directory=os.path.join(dir_manager.root_directory, "hello"), name="hello.py")

# Delete the folder
# dir_manager.delete_item(directory=dir_manager.root_directory, name="hello", is_folder=True)