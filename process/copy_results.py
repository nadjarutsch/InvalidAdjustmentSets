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
path_pattern = re.compile(r'/gpfs/scratch\d+/nodespecific/tcn\d+/nrutsch\.\d+/InvalidAdjustmentSets/results_\d+_estimated_\w+\.json')

# Read each .out file and find unique paths
for file in out_files:
    with open(file, 'r') as f:
        content = f.read()
        paths = path_pattern.findall(content)
        unique_paths.update(paths)

print(unique_paths)


# Copy each file to the local directory
for path in unique_paths:
    print(f"rsync -av nrutsch@snellius.surf.nl:{path} results_mbias \n")
#    path = path.strip()  # Remove any extra whitespace/newlines
#    subprocess.run(['scp', path, f"{local_user}@{local_host}:{local_directory}"])

