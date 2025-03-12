# %%
import pandas as pd

# %%
df = pd.read_excel('address_classification_all.xlsx')
df

# %%
df['classification'].value_counts()

# %%
