import glob
import re

# Define the directory containing the .out files
directory = "/gpfs/home5/nrutsch/InvalidAdjustmentSets/jobs/"

# Read all .out files
out_files = glob.glob(directory + "*.out")

# Initialize a set to store unique paths
unique_paths = set()

# Define the regex pattern
path_pattern = re.compile(r'\/gpfs\/scratch\d+\/nodespecific\/tcn\d+\/nrutsch\.\d+\/InvalidAdjustmentSets\/results_m2_\d+_estimated_\w+_\w+_optimality_.+\.json'
)

# Read each .out file and find unique paths
for file in out_files:
    with open(file, 'r') as f:
        content = f.read()
        paths = path_pattern.findall(content)
        unique_paths.update(paths)

print(unique_paths)

# Write the unique paths to a file
with open("files_to_rsync.txt", 'w') as f:
    for path in unique_paths:
        f.write(f"{path}\n")

# The rsync command:
print("rsync -av --files-from=files_to_rsync.txt nrutsch@snellius.surf.nl:/ results_m2/")
