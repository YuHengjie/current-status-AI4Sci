# %%
import os
import pandas as pd

# %%
# Define the directory containing the xls files
input_dir = os.path.join(os.getcwd(), "articles")
output_file = "article_combine.xlsx"

# Initialize an empty DataFrame to hold combined data
combined_data = pd.DataFrame()

# %%
# Loop through all files in the directory
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    # Check if the file has an xls extension
    if file_name.endswith(".xls"):
        # Read the content of the xls file
        df = pd.read_excel(file_path)
        # Append the data to the combined DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)

# One journal name may have upper term and lower term
combined_data['Source Title'] = combined_data['Source Title'].str.upper()
# Save the combined DataFrame to a new xlsx file
combined_data.to_excel(output_file, index=False)

print(f"Combined data has been saved to {output_file}")

# %%
