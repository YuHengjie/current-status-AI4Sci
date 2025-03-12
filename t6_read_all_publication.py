# %%
import os
import glob
import pandas as pd

# %%
# 设置文件夹路径
folder_path = 'articles_all_publication_number/'

# 获取所有txt文件路径
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
txt_files

# %%
# 准备收集数据的列表
data = []

# 遍历每个txt文件
for file_path in txt_files:
    # 提取期刊名（去除.txt扩展名）
    journal_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 初始化年份字典（2015-2024）
    year_counts = {str(year): 0 for year in range(2015, 2025)}
    year_counts['Journal'] = journal_name
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过标题行（假设第一行是标题）
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                year = parts[0]
                # 检查是否为有效年份
                if year.isdigit() and 2015 <= int(year) <= 2024:
                    # 处理可能存在的千分位分隔符
                    count = parts[1].replace(',', '')
                    if count.isdigit():
                        year_counts[year] = int(count)
    
    # 按正确年份顺序组织数据
    ordered_data = {
        'Journal': journal_name,
        **{str(year): year_counts[str(year)] for year in range(2015, 2025)}
    }
    data.append(ordered_data)
data

# %%
# 创建DataFrame
df = pd.DataFrame(data)

# 调整列顺序
columns = ['Journal'] + [str(year) for year in range(2015, 2025)]
df = df[columns]
df

# %%
# 新增汇总行部分
sum_row = df.select_dtypes(include='number').sum()
sum_row['Journal'] = 'Sum'
df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)
df

# %%
df.to_excel('all_publication_year.xlsx')
# %%
