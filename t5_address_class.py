# %%
from duckduckgo_search import DDGS
from openai import OpenAI
import pandas as pd
import os
import time

file_path = r"D:\api_key\deep_seek.txt"

# Load API Key
with open(file_path, "r") as file:
    api_key = file.read().strip()

# %% read address
address_df = pd.read_excel('address_classification.xlsx',index_col=0)
# address_df = address_df.iloc[0:30000,:]
address_df

# %%
# Ensure you have loaded your API key
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Iterate over each row in journal_abb
for idx, row in address_df.iterrows():
    
    # Check if 'classification' is not empty
    if not pd.isna(row['classification']) and str(row['classification']).strip() != '':
        continue  # Skip this row if 'classification' is not empty
    
    address = row['address']
    query = f"{address}"
    
    try:
        # Prepare the messages for the API
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", 
            "content": f"""##Task: Classify the following institution's address "{address}" into one of the following categories based on its primary research focus:
                            - 1. 'AI institution': Address related to artificial intelligence or informatics or data-driven research.
                            - 2. 'Science institution': Address related to natural science, health science, or non-AI engineering science.
                            - 3. 'Unclear institution': Address that does not clearly fit into the above two categories or contains both of them (such as advanced research institution and schoole of engineering).
                            
                            ##Institution's address: {address}
                            
                            ##Self reflection: Please reflect the initial classification on the following:
                            - 1. Does the address contain explicit keywords or references (e.g., 'AI', 'Artificial Intelligence', 'Machine Learning', 'Data Science') that align with the category 'AI institution'? 
                            - 2. Does the address mention terms related to natural sciences, health sciences, or traditional engineering disciplines that fit the category 'Science institution'? 
                            - 3. Is there ambiguity or a lack of clear indicators in the address that would make classification challenging, thus warranting the 'Unclear institution' category?
                            - 4. Reevaluate the decision by cross-checking with the provided categories and ensure no relevant information from the address has been overlooked. 
                            - 5. If confidence in the initial classification is low, would an alternative category be more appropriate?
                            
                            ##Output format: Please only provide the final category name as output, without any additional text or explanation."""}
        ]
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        # Extract the abbreviation from the response
        classification = response.choices[0].message.content.strip()
        
        # Assign the abbreviation to the 'abb' column
        address_df.at[idx, 'classification'] = classification
        
        print(f"Processed {idx}: {address}, Classification: {classification}")
    
    except Exception as e:
        print(f"Error processing {address}: {e}")
        continue
    
    
# %%
address_df.to_excel('address_classification.xlsx')

# %%
