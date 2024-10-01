# %% 

from pathlib import Path
import json 
import re
import yaml
# %%

def parse_tree_to_dict(tree):
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

            # Compute the level based on the number of '|   ' patterns
            level = bars.count('|   ')
            if match.group(3):  # If '|-- ' is present, increment level
                level += 1

            # Adjust the stack and levels to the current level
            while level <= levels[-1]:
                stack.pop()
                levels.pop()

            current_level = stack[-1]
            if name.startswith("/"):
                # It's a directory
                current_level[name] = {}
                stack.append(current_level[name])
                levels.append(level)
            else:
                # It's a file
                current_level[name] = ""
        else:
            # Root directory
            name = line.strip()
            root[name] = {}
            stack = [root[name]]
            levels = [0]

    return root

# %%
def parse_dict_to_tree(d):
    def helper(current_dict, prefix):
        lines = []
        keys = list(current_dict.keys())
        for idx, key in enumerate(keys):
            # Determine if it's the last item to adjust the prefix accordingly
            is_last = idx == len(keys) - 1
            connector = '|-- '
            line = prefix + connector + key
            lines.append(line)
            if isinstance(current_dict[key], dict) and current_dict[key]:
                # If not the last item, keep the '|' in the prefix
                extension = '|   ' if not is_last else '    '
                lines.extend(helper(current_dict[key], prefix + extension))
        return lines

    lines = []
    keys = list(d.keys())
    for idx, key in enumerate(keys):
        # Root nodes have no prefix
        lines.append(key)
        if isinstance(d[key], dict) and d[key]:
            lines.extend(helper(d[key], ''))
    return '\n'.join(lines)


# %% test 

tree_structure = """
/my-nextjs-project
|-- /components
|   |-- /auth
|   |   |-- LoginForm.js
|   |   |-- SignupForm.js
|   |   |-- AuthProvider.js
|   |-- /blog
|   |   |-- BlogPost.js
|   |   |-- BlogList.js
|   |-- Header.js
|   |-- Footer.js
|-- /pages
|   |-- /api
|   |   |-- /auth
|   |   |   |-- login.js
|   |   |   |-- signup.js
|   |   |   |-- logout.js
|   |   |-- /blog
|   |   |   |-- posts.js
|   |-- /auth
|   |   |-- login.js
|   |   |-- signup.js
|   |-- /blog
|   |   |-- [id].js
|   |   |-- index.js
|   |-- _app.js
|   |-- index.js
|-- /public
|   |-- /images
|-- /styles
|   |-- globals.css
|   |-- Home.module.css
|-- /utils
|   |-- auth.js
|   |-- api.js
|-- /hooks
|   |-- useAuth.js
|-- /context
|   |-- AuthContext.js
|-- /lib
|   |-- db.js
|-- .env.local
|-- next.config.js
|-- package.json
|-- README.md
"""


parsed_tree = parse_tree_to_dict(tree_structure)
print(parsed_tree)

parsed_dict = parse_dict_to_tree(parsed_tree)
print(parsed_dict)
# %%
