import os
import re

# Read the base path to remove from an environment variable
base_path = os.getenv('CARLIST_BASE_PATH', '')

if not base_path:
    raise ValueError("Environment variable 'CARLIST_BASE_PATH' is not set.")

# Read the contents of the file
with open("carlist.txt", "r") as file:
    lines = file.readlines()
    print(lines)

# Define the regex pattern to remove the specified part
pattern = re.escape(base_path)

# Apply the regex to each line
updated_lines = [re.sub(pattern, "", line) for line in lines]

# Save the updated lines back to the file
with open("carlist.txt", "w") as file:
    file.writelines(updated_lines)
