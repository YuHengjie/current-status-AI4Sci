# %%
import os
import glob

# %%
# Get the current working directory
current_directory = os.getcwd()

# Construct the path to the 'articles' subdirectory
articles_directory = os.path.join(current_directory, 'articles')

# Use glob to find all files ending with '.xls' in the 'articles' directory
xls_files = glob.glob(os.path.join(articles_directory, '*.xls'))

# Remove the '.xls' extension and keep only the file names
file_names = [os.path.splitext(os.path.basename(file))[0] for file in xls_files]

file_names

# %%
# Open and read the file
with open('nature index journal list.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Read all lines from the file

# Process each line: remove empty lines and strip the last parentheses
processed_lines = []
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespace (including newlines)
    if line:  # Check if the line is not empty
        # Find the position of the last opening parenthesis
        last_bracket_index = line.rfind('(')
        if last_bracket_index != -1:  # If a parenthesis is found
            line = line[:last_bracket_index].strip()  # Remove the last parenthesis and its content
        processed_lines.append(line)  # Add the processed line to the list

# Print the processed lines
for line in processed_lines:
    print(line)

# %%
# Convert both lists to sets
set_processed_lines = set(processed_lines)  # Convert processed_lines to a set
set_file_names = set(file_names)  # Convert file_names to a set

# Find elements in processed_lines but not in file_names
difference_processed_minus_file = set_processed_lines - set_file_names

# Find elements in file_names but not in processed_lines
difference_file_minus_processed = set_file_names - set_processed_lines

# Print the results
print("Elements in processed_lines but not in file_names:", difference_processed_minus_file)
print("Elements in file_names but not in processed_lines:", difference_file_minus_processed)

# %%
