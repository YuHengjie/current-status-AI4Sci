# %%
from duckduckgo_search import DDGS
from openai import OpenAI
import pandas as pd
import os

file_path = r"D:\api_key\deep_seek.txt"

# Load API Key
with open(file_path, "r") as file:
    api_key = file.read().strip()

# %% read journal name
data_df = pd.read_excel('article_combine.xlsx')
unique_journals = data_df['Source Title'].unique()
unique_journals

# %%
journal_abb = pd.DataFrame({
    'journal name': unique_journals,  
    'abb': None  
})
journal_abb

# %%
# Ensure you have loaded your API key
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# Initialize DDGS instance
ddgs = DDGS()

# Create the directory for saving files
output_dir = 'web_search_abb'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Iterate over each row in journal_abb
for idx, row in journal_abb.iterrows():
    journal_name = row['journal name']
    
    # Construct the search query
    query = f"Abbreviation for journal title {journal_name}"
    
    try:
        # Perform web search
        results = ddgs.text(query, max_results=5)
        
        # Extract titles and bodies from the results
        search_context = "\n\n".join([f"Title {i+1}: {result['title']}\nBody {i+1}: {result['body']}" 
                                       for i, result in enumerate(results)])
        
        # Create the context string
        context = f"##Context from web search results:\n{search_context}\n\n"
        
        # Replace spaces with underscores for the filename
        safe_journal_name = journal_name.replace(' ', '_')
        filename = f'search_result_for_{safe_journal_name}.txt'
        file_path = os.path.join(output_dir, filename)
        # Save the context to a text file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(context)
            
        # Prepare the messages for the API
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", 
             "content": f"""##Question: What is the abbreviation of the journal title {journal_name}? You can answer based on the following web search results.
                            {context}
                            ##Output format: Please only input the abbreviation of the journal name, do not output any extra content."""}
        ]
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        # Extract the abbreviation from the response
        abbreviation = response.choices[0].message.content.strip()
        
        # Assign the abbreviation to the 'abb' column
        journal_abb.at[idx, 'abb'] = abbreviation
        
        print(f"Processed {journal_name}, Abbreviation: {abbreviation}")
    
    except Exception as e:
        print(f"Error processing {journal_name}: {e}")
        continue

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
