import glob
import re
import subprocess

# Define the directory containing the .out files
directory = "/gpfs/home5/nrutsch/InvalidAdjustmentSets/jobs/"

# Read all .out files
out_files = glob.glob(directory + "*.out")

# Initialize a set to store unique paths
unique_paths = set()

# Define the regex pattern
path_pattern = re.compile(r'/gpfs/scratch\d+/nodespecific/tcn\d+/nrutsch\.\d+/InvalidAdjustmentSets/results_\d+_estimated_\w+\$')

# Read each .out file and find unique paths
for file in out_files:
    with open(file, 'r') as f:
        content = f.read()
        print(content)
        paths = path_pattern.findall(content)
        unique_paths.update(paths)

print(unique_paths)

# Define the local directory to save the files
local_user = "nadrut"
local_host = "Nadjas-MacBook-Pro.local"
local_directory = "/Users/nadrut/PycharmProjects/nO_examples/results_mbias"  # Update with your local path

# Copy each file to the local directory
for path in unique_paths:
    path = path.strip()  # Remove any extra whitespace/newlines
    subprocess.run(['scp', path, f"{local_user}@{local_host}:{local_directory}"])

