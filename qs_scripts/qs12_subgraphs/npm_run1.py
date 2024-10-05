

# read project_params json 
import json 
with open("project_params.json", "r") as f:
    project_params = json.load(f)
project_name = project_params['nextjs']['project_name']


import subprocess
import os

# Step 1: Define the project name and directory
project_dir = os.path.join(os.getcwd(), project_name)

# Step 2: Define the flags to predefine answers
flags = [
    '--typescript',
    '--eslint',
    '--no-src-dir',
    '--app',
    '--tailwind',
    '--import-alias', '@/*',
    '--use-npm'
]

# Step 3: Check if the project directory already exists
if not os.path.exists(project_dir):
    # Project does not exist, create a new one
    command = ["npx", "create-next-app@latest", project_name] + flags
    try:
        print(f"Creating a Next.js project: {project_name}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating Next.js project: {e}")
        exit(1)
else:
    print(f"Project '{project_name}' already exists. Skipping project creation.")

# Step 4: Navigate to the project directory
os.chdir(project_dir)

# Step 5: Start the development server using 'npm run dev'
try:
    print(f"Starting the Next.js development server in: {project_dir}")
    subprocess.run(["npm", "run", "dev"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error starting Next.js development server: {e}")