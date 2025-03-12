# %%
import pandas as pd
import numpy as np

# 读取Excel文件（注意列名包含Journal和Abb两个独立列）
df_ai = pd.read_excel('ai_publication_year_with_abb.xlsx')
df_all = pd.read_excel('all_publication_year_with_abb.xlsx')
df_all

# %%
# 按期刊缩写分组求和（自动排除非数值列Journal）
#df_ai = df_ai.groupby('Abb', as_index=False).sum()# 直接修改原DataFrame
df_ai.drop('Journal', axis=1, inplace=True)
#df_all = df_all.groupby('Abb', as_index=False).sum()
df_all.drop('Journal', axis=1, inplace=True)
df_all

# %%
# 设置索引用于对齐计算
df_ai.set_index('Abb', inplace=True)
df_all.set_index('Abb', inplace=True)

# 计算百分比（自动对齐相同的Abb和年份列）
percentage_df = (df_ai / df_all) * 100

# 处理异常值
percentage_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 重置索引并整理列顺序
result = percentage_df.reset_index()
result = result[['Abb'] + [str(y) for y in range(2015, 2025)]]

result

# %%
result.to_excel('journal_year_percent.xlsx')

# %%
