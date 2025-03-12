# %%
import pandas as pd
import re
from tqdm import tqdm

# %%
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
# 定义一个函数来决定是否替换逗号为分号
def replace_commas_if_needed(text):
    # 计算逗号的数量
    comma_count = text.count(',')
    if comma_count > 3:
        # 如果逗号数量大于3，则将所有逗号替换为分号
        return text.replace(',', ';')
    else:
        # 否则保持原样
        return text

# 应用该函数到 'Terms' 列上
df['Terms'] = df['Terms'].apply(replace_commas_if_needed)

# 检查 'Terms' 列中每个值，如果不存在 ';' 则将其中的 ',' 替换成 ';'
df['Terms'] = df['Terms'].apply(lambda x: x.replace(',', ';') if ';' not in x else x)


# %%
# 拆分术语，并展开为单独的行
all_terms = df['Terms'].str.split(';').explode()
# 使用 .str.strip() 方法去除每个字符串前后的多余空格
all_terms = all_terms.str.strip()
# 定义一个函数来去除字符串前后的标点符号
def remove_punctuation(text):
    # 使用正则表达式去除字符串开头和结尾的标点符号
    return re.sub(r'^\W+|\W+$', '', text)

# 应用该函数到 Series 上
all_terms = all_terms.apply(remove_punctuation)
print(len(all_terms))

# %%
def process_parentheses(text):
    # 检查是否包含 '('
    if '(' in text:
        # 获取 '(' 位置
        start_index = text.index('(')
        # 检查 '(' 后面的字符数
        if len(text[start_index:]) < 5 + 1:  # 加1是因为要包括 '(' 本身
            # 如果 '(' 后面的字符数（包括 '('）小于6，则移除 '(' 及其后面的所有内容
            return text[:start_index]
        else:
            return text
    else:
        return text

# 应用该函数到 Series 上
all_terms = all_terms.apply(process_parentheses)
all_terms

# %%
# 统计每个术语的出现次数
term_counts = all_terms.value_counts().reset_index()
term_counts.columns = ['Term', 'Count']
term_counts

# %%
# 自定义函数仅将字符串的首字符转换为大写
def capitalize_first_letter(s):
    return s[:1].upper() + s[1:] if s else ''

# 应用此函数到 'Term' 列上
term_counts['Term'] = term_counts['Term'].apply(capitalize_first_letter)
term_counts

# %%
# 归一化术语（转换为小写）
term_counts['Lower_Term'] = term_counts['Term'].str.lower()

# 按小写术语合并，保留第一个原始术语
term_counts = term_counts.groupby('Lower_Term', as_index=False).agg({
    'Term': 'first',  # 保留第一个出现的原始术语
    'Count': 'sum'    # 统计所有大小写变体的总次数
})

# 删除辅助列
term_counts = term_counts.drop(columns=['Lower_Term'])
# **按 Count 降序排序**
term_counts = term_counts.sort_values(by='Count', ascending=False)
term_counts

# %%
term_counts = term_counts[term_counts['Term'].apply(lambda x: isinstance(x, str))]
# 使用 apply 方法和 len 函数计算字符串长度，并将结果存入新列 'Term_Length'
term_counts['Term_Length'] = term_counts['Term'].apply(len)
term_counts = term_counts.sort_values(by='Term_Length', ascending=False)
term_counts

# %%
# 筛选出'Term_Length'列中值大于4且小于50的行
filtered_terms = term_counts[(term_counts['Term_Length'] > 4) & (term_counts['Term_Length'] < 40)]
filtered_terms

# %%
# 删除'Term_Length'列
if 'Term_Length' in filtered_terms.columns:
    filtered_terms = filtered_terms.drop(columns=['Term_Length'])

# 按照'Count'列从大到小排序
filtered_terms_sorted = filtered_terms.sort_values(by='Count', ascending=False)
filtered_terms_sorted

# %%
clean_terms = filtered_terms_sorted.copy()
clean_terms['Term'] = clean_terms['Term'].apply(lambda x: ' '.join(word.capitalize() for word in x.split()))
clean_terms

# %%
clean_terms = clean_terms[clean_terms['Count'] >= 2]
clean_terms

# %%
# 重置index，并丢弃原有的index
clean_terms = clean_terms.reset_index(drop=True)
clean_terms

# %%
# 假设 clean_terms 是一个包含 'Term' 和 'Count' 两列的 DataFrame
def normalize_text(text):
    """
    标准化文本：转小写并移除标点符号、所有形式的连字符（-、–、—）以及空格
    """
    text = text.lower()  # 转换为小写
    # 移除标点符号、所有形式的连字符及空白字符
    return re.sub(r'[^\w]|[-–—]', '', text)  # 此处移除了所有非单词字符和连字符，包括空格
# 创建一个空字典来存储结果
similar_terms_dict = {}

# 使用 tqdm 包装 iterrows() 迭代器以显示进度条
for index, row in tqdm(clean_terms.iterrows(), total=clean_terms.shape[0], desc="Processing terms"):
    raw_term = row['Term']
    term = normalize_text(raw_term)  # 标准化处理
    
    if term == 'human':
        continue
    
    added_to_dict = False
    for key in list(similar_terms_dict.keys()):
        if term.startswith(normalize_text(key)):
            # 如果当前term以某个key为前缀，添加到对应value列表
            similar_terms_dict[key].append(row['Term'])
            added_to_dict = True
            break  # 找到匹配后退出循环避免重复添加
    
    if not added_to_dict:
        # 若没有找到合适的key，则将当前term设置为新key
        similar_terms_dict[raw_term] = [raw_term]
# 打印结果查看
similar_terms_dict

# %%
# 初始化空列表用于存储新DataFrame的数据
data = []

# 使用tqdm显示进度条
for key, similar_terms in tqdm(similar_terms_dict.items(), desc="Clustering terms"):
    # 计算相似词组的count总和
    total_count = clean_terms[clean_terms['Term'].isin(similar_terms)]['Count'].sum()
    
    # 将信息添加到data列表中
    data.append({
        'Term': key,
        'Similar terms': '; '.join(similar_terms),  # 将列表转换为;分隔的字符串
        'Count': total_count
    })

# 创建新的DataFrame
cluster_terms = pd.DataFrame(data)
cluster_terms

# %%
# 按照'Count'列从大到小排序
cluster_terms = cluster_terms.sort_values(by='Count', ascending=False)
cluster_terms

# %%
cluster_terms.to_excel('term_counts_cluster.xlsx',index=False)

# %%
