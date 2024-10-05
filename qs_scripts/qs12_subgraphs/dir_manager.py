import os
import re
import shutil
import json

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
        full_path = os.path.join(directory, name)
        if is_folder:
            os.makedirs(full_path, exist_ok=True)
            print(f"Folder created: {full_path}")
        else:
            os.makedirs(directory, exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content or "")
            print(f"File created: {full_path}")

    def overwrite_file(self, directory, name, content):
        file_path = os.path.join(directory, name)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"File overwritten: {file_path}")

    def delete_item(self, directory, name, is_folder=False):
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

    def get_tree_structure(self):
        tree_dict = self.folder_tree_dict()
        return self.parse_dict_to_tree(tree_dict)
    
# Assuming 'project_params.json' is in the current directory and contains the necessary data

# Initialize DirManager with the path to 'project_params.json'
dir_manager = DirManager(project_params_path="project_params.json")

# Print the project name
print(f"Project Name: {dir_manager.project_name}")

# Get the current tree structure
print(dir_manager.get_tree_structure())

# Add a new folder inside the project directory
# dir_manager.add_to_directory(directory=dir_manager.root_directory, name="hello", is_folder=True)

# Add a new file inside the new folder
# dir_manager.add_to_directory(directory=os.path.join(dir_manager.root_directory, "hello"), name="hello.py", content="print('Hello, World!')")

# Overwrite the existing file
# dir_manager.overwrite_file(directory=os.path.join(dir_manager.root_directory, "hello"), name="hello.py", content="print('Hello, Universe!')")

# Delete the file
# dir_manager.delete_item(directory=os.path.join(dir_manager.root_directory, "hello"), name="hello.py")

# Delete the folder
# dir_manager.delete_item(directory=dir_manager.root_directory, name="hello", is_folder=True)