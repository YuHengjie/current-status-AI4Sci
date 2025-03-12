# %%
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %%
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
# 定义一个函数用于计算字符串中的字母数量
def count_letters(s):
    if isinstance(s, str):
        # 使用正则表达式匹配字符串中的所有字母并计算其数量
        return len(re.findall(r'[a-zA-Z]', s))
    return 0

# 确保'Terms'列中的所有元素都是字符串，如果不是字符串则标记为True
not_string_mask = df['Terms'].apply(lambda x: not isinstance(x, str))

# 找出是字符串且包含字母数少于50个的行
short_terms_mask = df['Terms'].apply(lambda x: count_letters(x) < 100)

# 结合两个mask，找到符合条件的所有行
final_mask = not_string_mask | short_terms_mask

# 使用最终的mask筛选出需要的行
filtered_df = df[final_mask]
filtered_df

# %%
# 找到不包含分号";"的所有行
missing_semicolon = df[~df['Terms'].str.contains(';', na=False)]
missing_semicolon

# %%
# 找到分号";"数量少于2的所有行
missing_semicolon_2 = df[df['Terms'].apply(lambda x: str(x).count(';') < 2)]
missing_semicolon_2

# %%
# 初始化模型和分词器
model_path = r"D:\PreTrainedModels\Llama3-KALE-LM-Chem-1.5-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# %%
def clean_terms_with_llm(raw_str):
    """使用LLM清理并标准化术语分隔"""
    # 构造明确格式要求的prompt
    prompt = f"""Task: Clean and standardize the formatting of scientific terms
    
Input string: "{raw_str}"

Requirements:
1. Split compound terms into individual items
2. Use SEMICOLONS (;) as separators
3. Keep original capitalization and hyphens
4. Remove redundant spaces
5. Output format example: "Term1; Term2; Term3"

Respond ONLY with the cleaned string and DO NOT output with any additional comments or explanations"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取清理后的结果
    cleaned = response.split("cleaned string:")[-1].strip().strip('"')
    
    # 后处理确保分号格式
    cleaned = re.sub(r'\s*;\s*', '; ', cleaned)  # 统一分号间距
    cleaned = re.sub(r';\s*$', '', cleaned)      # 去除末尾分号
    return cleaned

# %% test
raw_str = missing_semicolon.iloc[0,-1]
raw_str

# %%
cleaned = clean_terms_with_llm(raw_str)
cleaned

# %%
# 处理并添加新列
df['Clear terms'] = df['Terms'].apply(lambda x: 
    x if (pd.notna(x) and x.count(';') >= 2) 
    else clean_terms_with_llm(x) if pd.notna(x) 
    else ''
)

# %%
# 创建新的列用于存储清理后的terms
df['Clear terms'] = None

# 遍历DataFrame的每一行进行处理
for idx, row in df.iterrows():
    
    # print(f"Processing {idx}")
    raw_str = row['Terms']
    
    if raw_str.count(';') >= 2:
        # 如果已经有足够的分号，直接复制原字符串
        df.at[idx, 'Clear terms'] = raw_str
        continue
    
    
    """使用LLM清理并标准化术语分隔"""
    # 构造明确格式要求的prompt
    prompt = f"""#Task: Clean and standardize the formatting of scientific terms
    
#Input string: "{raw_str}"

#Important Notes:
- Each term is a complete phrase with scientific meaning, not just a single word.
- The input string may contain various delimiters such as commas (,), periods (.), or numbers followed by periods (e.g., "1.", "2.") as separators. These should be replaced with semicolons (;) as part of the cleaning process.

#Requirements:
1. Split compound terms into individual items based on their scientific meaning. Do not split phrases into individual words.
2. Use SEMICOLONS (;) exclusively as separators between terms.
3. Maintain original capitalization and hyphens within each term.
4. Remove redundant spaces around terms and separators.
5. Output format example: "Term1; Term2; Term3"

#Output format:
Respond ONLY with the cleaned string without any additional comments or explanations. 
For example: Term1; Term2; Term3"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Processing {idx}: {raw_str}/n Cleaned: {response}")

    del inputs, outputs, response
    torch.cuda.empty_cache()
    
# %%
# 保存结果
df.to_excel('scientific_terms_cleaned.xlsx', index=False)