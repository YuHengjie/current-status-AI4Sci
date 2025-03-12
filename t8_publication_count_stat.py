# %%
import pandas as pd

# %%
# 读取名称匹配关系表
name_pairs = pd.read_excel('name_pair.xlsx.xlsx')

# 创建期刊名称到缩写的映射字典（处理多列名称情况）
name_columns = [col for col in name_pairs.columns if col != 'abb']
abb_dict = {}
for _, row in name_pairs.iterrows():
    abb = row['abb']
    for col in name_columns:
        name = row[col]
        if pd.notna(name):
            abb_dict[name] = abb

# 读取目标文件并添加缩写列
df = pd.read_excel('ai_publication_year.xlsx')
df['Abb'] = df['Journal'].map(abb_dict)

# 调整列顺序：将abb列移动到Journal列之后
cols = df.columns.tolist()          # 获取所有列名
cols.remove('Abb')                  # 移除abb列
cols.insert(1, 'Abb')               # 在位置1（Journal列索引为0，其后插入）
df = df[cols]                       # 重组列顺序

# 保存结果到新文件
df.to_excel('ai_publication_year_with_abb.xlsx', index=False)

# %%
# 读取目标文件并添加缩写列
df = pd.read_excel('all_publication_year.xlsx',index_col=0)
df['Abb'] = df['Journal'].map(abb_dict)

# 调整列顺序：将abb列移动到Journal列之后
cols = df.columns.tolist()          # 获取所有列名
cols.remove('Abb')                  # 移除abb列
cols.insert(1, 'Abb')               # 在位置1（Journal列索引为0，其后插入）
df = df[cols]                       # 重组列顺序

# 保存结果到新文件
df.to_excel('all_publication_year_with_abb.xlsx', index=False)

# %%
