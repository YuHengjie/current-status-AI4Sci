# %% test duckduckgo_search
from duckduckgo_search import DDGS
import http.client
import urllib.parse
import requests

file_path = r"D:\api_key\duckduckgo.txt"

# Load duckduckgo API Key
with open(file_path, "r") as file:
    duckduckgo_api_key = file.read().strip()

# %%
headers = {
    'x-rapidapi-key': duckduckgo_api_key,
    'x-rapidapi-host': "duckduckgo-search-api.p.rapidapi.com"
}

# Define your search query
query = "What is the capital of Canada?"
# Encode the query parameter to handle special characters
encoded_query = urllib.parse.quote(query)

# Construct the request URL with the query parameter
url = f"/htmlSearch?q={encoded_query}&df=d&kl=ar-es"

# Create a connection to the RapidAPI endpoint
conn = http.client.HTTPSConnection("duckduckgo-search-api.p.rapidapi.com")

# Send the GET request
conn.request("GET", url, headers=headers)

# Get the response
res = conn.getresponse()
data = res.read()

# Print the response data
print(data.decode("utf-8"))

# %%
url = "https://duckduckgo.com"

querystring = {"q":"apple","df":"d","kl":"ar-es"}

headers = {
	"x-rapidapi-key": duckduckgo_api_key,
	"x-rapidapi-host": "duckduckgo-search-api.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())

# %% free
# Define your search query
query = "What is the capital of Canada?"
query = "Northeastern Univ, Dept Mech & Ind Engn, Boston, MA 02115 USA"
# Create a DDGS instance
ddgs = DDGS()

# Get search results
results = ddgs.text(query, max_results=5)

# %%
# Extract titles and URLs from the results
search_context = "\n\n".join([f"Title {idx}: {result['title']}\nBody {idx}: {result['body']}" 
                               for idx, result in enumerate(results, start=1)])
# Create a context string
context = f"##Context from web search results:\n{search_context}\n\n"

print(context)

# %% test deepseek
from openai import OpenAI

file_path = r"D:\api_key\deep_seek.txt"

# Load API Key
with open(file_path, "r") as file:
    api_key = file.read().strip()


# %%
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)