# %%
from duckduckgo_search import DDGS
from openai import OpenAI
import pandas as pd
import os

file_path = r"D:\api_key\deep_seek.txt"

# Load API Key
with open(file_path, "r") as file:
    api_key = file.read().strip()

# %% read addresses
data_df = pd.read_excel('article_combine.xlsx')
address_df = data_df[['DOI', 'Publication Year', 'Source Title', 'Article Title', 'Addresses']]
address_df

# %%
# Remove text within square brackets and the brackets themselves
address_df['Addresses_edit'] = address_df['Addresses'].str.replace(r'\[.*?\]', '', regex=True)

# Replace multiple spaces with a single space and strip leading/trailing spaces
address_df['Addresses_edit'] = address_df['Addresses_edit'].str.replace(r'\s+', ' ', regex=True).str.strip()
address_df

# %%
# Define a function to calculate the address number
def calculate_address_number(addresses):
    if pd.isna(addresses):
        return 1  # or any default value you prefer for NaN cases
    addresses_list = addresses.split('; ')
    # Filter out any empty strings resulting from split
    addresses_list = [addr for addr in addresses_list if addr.strip()]
    return len(addresses_list) + 1

# Apply the function to create the new column
address_df['address number'] = address_df['Addresses_edit'].apply(calculate_address_number)
address_df

# %%
# Initialize an empty list to hold all addresses
all_addresses = []

# Iterate over each value in the 'Addresses_edit' column
for address in address_df['Addresses_edit']:
    if pd.isna(address):
        continue  # Skip NaN values
    if ';' in address:
        # Split the string by '; ' and extend the list with non-empty substrings
        substrings = [substr.strip() for substr in address.split(';') if substr.strip()]
        all_addresses.extend(substrings)
    else:
        # Append the string directly if no semicolon is present
        all_addresses.append(address.strip())
print(len(all_addresses))

# %% obtain unique address
seen_addresses = set()
unique_addresses = []

for address in all_addresses:
    if address not in seen_addresses:
        seen_addresses.add(address)
        unique_addresses.append(address)

print(len(unique_addresses))

# %%
# Create a DataFrame with the unique addresses and an empty classification column
address_classification_df = pd.DataFrame({
    'address': unique_addresses,
    'classification': None  # Initialize classification column with None (or use "" for empty strings)
})
address_classification_df

# %%
address_classification_df.to_excel('address_classification_raw.xlsx')

# %%
