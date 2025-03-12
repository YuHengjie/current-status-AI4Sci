# %%
import os
import pandas as pd

# %%
# Define the path to the 'articles' directory
articles_dir = 'articles'

# Loop through all files in the 'articles' directory
for filename in os.listdir(articles_dir):
    # Construct the full file path
    file_path = os.path.join(articles_dir, filename)
    
    try:
        # Read the Excel file with the appropriate engine for .xls files
        df = pd.read_excel(file_path, engine='xlrd')
        
        # Get the number of rows and columns
        num_rows, num_columns = df.shape
        
        # Check if the number of columns is not equal to 72
        if num_columns != 72:
            print(f"{filename}: Number of columns is not equal to 72. It has {num_columns} columns.")
        
        # Check if the number of rows is either 50 or 1000
        if num_rows == 50 or num_rows == 1000:
            print(f"{filename}: Number of rows is equal to {num_rows}.")
    
    except Exception as e:
        print(f"Error reading {filename}: {e}")
            
# %%
