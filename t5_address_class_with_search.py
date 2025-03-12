# %%
from duckduckgo_search import DDGS
from openai import OpenAI
import pandas as pd
import os
import time
import http.client

file_path = r"D:\api_key\deep_seek.txt"

# Load deepseek API Key
with open(file_path, "r") as file:
    deepseek_api_key = file.read().strip()
    
file_path = r"D:\api_key\duckduckgo.txt"

# Load duckduckgo API Key
with open(file_path, "r") as file:
    duckduckgo_api_key = file.read().strip()

# %% read journal name
address_df = pd.read_excel('address_classification.xlsx',index_col=0)
address_df

# %%
# Ensure you have loaded your API key
client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Initialize DDGS instance
ddgs = DDGS()

# Create the directory for saving files
output_dir = 'web_search_address'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
max_retries = 5

# Iterate over each row in journal_abb
for idx, row in address_df.iterrows():
    
    # Check if 'classification' is not empty
    if not pd.isna(row['classification']) and str(row['classification']).strip() != '':
        continue  # Skip this row if 'classification' is not empty
    
    address = row['address']
    query = f"{address}"
    
    # Construct the search query
    retry_count = 0
    search_successful = False
    while retry_count < max_retries:
        try:
            # Perform web search
            results = ddgs.text(query, max_results=5)
            
            # Extract titles and bodies from the results
            search_context = "\n\n".join([f"Title {i+1}: {result['title']}\nBody {i+1}: {result['body']}" 
                                            for i, result in enumerate(results)])
            
            # Create the context string
            context = f"##Context from web search results for reference:\n{search_context}\n\n"
        
            # Replace spaces with underscores for the filename
            safe_address_name = address.replace(' ', '_')
            filename = f'search_results_for_{safe_address_name}.txt'
            file_path = os.path.join(output_dir, filename)
            # Save the context to a text file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(context)
            
            search_successful = True
            break  # Exit the retry loop on success
            
        except Exception as e:
            wait_time = 2 ** retry_count  # Exponential backoff
            print(f"--------Retry. Waiting for {wait_time} seconds. Error: {e}")
            time.sleep(wait_time)
            retry_count += 1
            
    if not search_successful:
        # Log the failure
        print(f"----Failed to retrieve search results for address: {address}")
        # Skip classification for this address
        continue  # Proceed to the next iteration
    
    
    # Prepare the messages for the API
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", 
        "content": f"""##Task: Classify the following institution's address "{address}" into one of the following categories based on its primary research focus:
                        - 1. 'AI institution': Address related to artificial intelligence or data-driven research.
                        - 2. 'Science institution': Address related to natural science, health science, or non-AI engineering science.
                        - 3. 'Unclear institution': Address that does not clearly fit into the above two categories.
                        
                        ##Institution's address: {address}
                        
                        {context}
                        
                        ##Self reflection: Please reflect the initial classification on the following:
                        - 1. Does the address contain explicit keywords or references (e.g., 'AI', 'Artificial Intelligence', 'Machine Learning', 'Data Science') that align with the category 'AI institution'? 
                        - 2. Does the address mention terms related to natural sciences, health sciences, or traditional engineering disciplines that fit the category 'Science institution'? 
                        - 3. Is there ambiguity or a lack of clear indicators in the address that would make classification challenging, thus warranting the 'Unclear institution' category?
                        - 4. Reevaluate the decision by cross-checking with the provided categories and ensure no relevant information from the address or context has been overlooked. 
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
    address_df.at[idx, 'abb'] = classification
    
    print(f"Processed {idx}: {address}, Classification: {classification}")
    
    #except Exception as e:
    #    print(f"Error processing {address}: {e}")
    #    continue

# %%
journal_abb_edit = journal_abb.copy()
# Remove all periods from the 'abb' column
journal_abb_edit['abb'] = journal_abb_edit['abb'].str.replace('.', '', regex=False)

# Convert to title case if the string is all uppercase
journal_abb_edit.loc[journal_abb_edit['abb'].str.isupper(), 'abb'] = \
    journal_abb_edit.loc[journal_abb_edit['abb'].str.isupper(), 'abb'].str.title()

journal_abb_edit

# %%
journal_abb_edit.to_excel('journal_abb_edit.xlsx')
# %%
