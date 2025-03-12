# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
import string

# %% read address
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
# 找到不包含分号";"的所有行
missing_semicolon = df[~df['Terms'].str.contains(';', na=False)]
missing_semicolon

# %%
# 检查是否包含分号，并标记需要更新的行
need_update = ~df['Terms'].str.contains(';', na=False)

# 更新条件：当逗号数量大于三个时执行替换
def replace_commas_if_exceeds_threshold(terms_str, threshold=5):
    if terms_str.count(',') > threshold:
        return terms_str.replace(',', '; ')
    return terms_str

# 应用替换逻辑到需要更新的行
df.loc[need_update, 'Terms'] = df.loc[need_update, 'Terms'].apply(lambda x: replace_commas_if_exceeds_threshold(x))
df

# %%# 找到不包含分号";"的所有行
missing_semicolon = df[~df['Terms'].str.contains(';', na=False)]
missing_semicolon


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
# 加载本地模型和分词器
model_path = "../pretrained_model/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# 将模型移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# %%
def extract_last_non_empty_line(text):
    # 将文本按换行符分割成多行
    lines = text.split('\n')
    
    # 从最后一行开始向上查找，直到找到非空行
    for line in reversed(lines):
        if line.strip():  # 如果该行非空
            return line.strip()
    return None  # 如果没有找到非空行，返回None

# %%
count_index = 0
# Iterate over each row in journal_abb
for idx, row in df.iterrows():
    
    title = row['Article Title']
    abstract = row['Abstract']
    reasoning_output = row['Reasoning']
    terms = row['Terms']

    # 如果terms是字符串并且字母数量大于100，则跳过本次循环
    if isinstance(terms, str) and count_letters(terms) > 100:
        continue  # 跳过此次迭代
    
    try:
        prompt  = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
#Task: Extract scientifically meaningful key terms from the given academic TITLE and ABSTRACT, following these rules:
# 1. Identify technical terms representing distinct scientific concepts/methods/materials
# 2. Capitalize only the first letter of the first word in the extracted term unless specific academic conventions dictate otherwise (e.g., proper nouns, established formulas, or field-specific norms)
# 3. Follow the common rules for using hyphens for specific terms in academia
# 4. Retain full original terminology if the abbreviation is not very well-known
# 5. Exclude generic prepositions/articles (of/the/for/etc.)
# 6. Prioritize multi-word technical phrases over single words
# 7. Ensure the output terms are concise and avoid repetition or similar vocabulary, outputting each unique term only once

#TITLE:{title}
#ABSTRACT:{abstract}

#Output format: ONLY output extracted terms in plain text (without any bolding, line breaking, or numbering) and DO Not output any unnecessary content. Use semicolons to separate the extracted academic terms"""

        # **Tokenize 并生成输出**
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                           truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保数据在 GPU/CPU 上

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, 
                                    top_k=50, top_p=0.9, temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id)

        # **解码输出**
        reasoning_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # **存入 DataFrame**
        df.at[idx, 'Reasoning'] = reasoning_output
        extracted_terms = extract_last_non_empty_line(reasoning_output)
        df.at[idx, 'Terms'] = extracted_terms
        print(f"Processed {idx}: {title}, Terms: {extracted_terms}")

        count_index += 1
        # 保存逻辑
        if (count_index + 1) % 50 == 0:  # 每处理50条保存一次
            df.to_excel('scientific_terms_all_last.xlsx')
            
        # **清理 GPU 内存**
        del inputs, output, prompt, reasoning_output  # 删除变量
        torch.cuda.empty_cache()  # 释放显存
             
    except Exception as e:
        print(f"Error processing {title}: {e}")


# %%
# %% read address
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
def has_word_repeated_more_than_n_times(terms_str, threshold=5):
    """
    判断给定的字符串中是否有单词重复超过指定的阈值次数。
    
    参数:
    - terms_str: 字符串，术语列表，以空格分隔。
    - threshold: 整数，判断单词是否重复过多的阈值。
    
    返回:
    - Boolean: 如果有任何单词重复超过阈值次，则返回True；否则返回False。
    """
    # 移除标点符号
    terms_str = terms_str.translate(str.maketrans('', '', string.punctuation))
    # 使用空格分割术语并清理空格
    terms_list = [term.lower().strip() for term in terms_str.split(' ') if term.strip()] # 转换为小写以便正确计数
    # 计算每个单词的频率
    word_counts = pd.Series(terms_list).value_counts()
    # 检查是否有单词出现超过阈值次
    return any(word_counts > threshold)

# %%
count_index = 0
# Iterate over each row in journal_abb
for idx, row in df.iterrows():
    
    title = row['Article Title']
    abstract = row['Abstract']
    reasoning_output = row['Reasoning']
    terms = row['Terms']

    # 如果单词重复次数小于5
    if not has_word_repeated_more_than_n_times(terms,threshold=5):
        continue  # 跳过此次迭代
    
    try:
        prompt  = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
#Task: Extract scientifically meaningful key terms from the given academic TITLE and ABSTRACT, following these rules:
# 1. Identify technical terms representing distinct scientific concepts/methods/materials
# 2. Capitalize only the first letter of the first word in the extracted term unless specific academic conventions dictate otherwise (e.g., proper nouns, established formulas, or field-specific norms)
# 3. Follow the common rules for using hyphens for specific terms in academia
# 4. Retain full original terminology if the abbreviation is not very well-known
# 5. Exclude generic prepositions/articles (of/the/for/etc.)
# 6. Prioritize multi-word technical phrases over single words
# 7. Ensure the output terms are concise and avoid repetition or similar vocabulary, outputting each unique term only once

#TITLE:{title}
#ABSTRACT:{abstract}

#Output format: ONLY output extracted terms in plain text (without any bolding, line breaking, or numbering) and DO Not output any unnecessary content. Use semicolons to separate the extracted academic terms"""

        # **Tokenize 并生成输出**
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                           truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保数据在 GPU/CPU 上

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, 
                                    top_k=50, top_p=0.9, temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id)

        # **解码输出**
        reasoning_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # **存入 DataFrame**
        df.at[idx, 'Reasoning'] = reasoning_output
        extracted_terms = extract_last_non_empty_line(reasoning_output)
        df.at[idx, 'Terms'] = extracted_terms
        print(f"Processed {idx}: {title}, Terms: {extracted_terms}")

        count_index += 1
        # 保存逻辑
        if (count_index + 1) % 50 == 0:  # 每处理50条保存一次
            df.to_excel('scientific_terms_all_last.xlsx')
            
        # **清理 GPU 内存**
        del inputs, output, prompt, reasoning_output  # 删除变量
        torch.cuda.empty_cache()  # 释放显存
             
    except Exception as e:
        print(f"Error processing {title}: {e}")

# %%
df.to_excel('scientific_terms_all_last.xlsx')

# %%

# %% read address
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
count_index = 0
# Iterate over each row in journal_abb
for idx, row in df.iterrows():
    
    title = row['Article Title']
    abstract = row['Abstract']
    reasoning_output = row['Reasoning']
    terms = row['Terms']

    # 如果有分号
    if ';' in terms:
        continue  # 跳过此次迭代
    
    try:
        prompt  = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
#Task: Extract scientifically meaningful key terms from the given academic TITLE and ABSTRACT, following these rules:
# 1. Identify technical terms representing distinct scientific concepts/methods/materials
# 2. Capitalize only the first letter of the first word in the extracted term unless specific academic conventions dictate otherwise (e.g., proper nouns, established formulas, or field-specific norms)
# 3. Follow the common rules for using hyphens for specific terms in academia
# 4. Retain full original terminology if the abbreviation is not very well-known
# 5. Exclude generic prepositions/articles (of/the/for/etc.)
# 6. Prioritize multi-word technical phrases over single words
# 7. Ensure the output terms are concise and avoid repetition or similar vocabulary, outputting each unique term only once

#TITLE:{title}
#ABSTRACT:{abstract}

#Output format: ONLY output extracted terms in plain text (without any bolding, line breaking, or numbering) and DO Not output any unnecessary content. Use semicolons to separate the extracted academic terms"""

        # **Tokenize 并生成输出**
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                           truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保数据在 GPU/CPU 上

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, 
                                    top_k=50, top_p=0.9, temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id)

        # **解码输出**
        reasoning_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # **存入 DataFrame**
        df.at[idx, 'Reasoning'] = reasoning_output
        extracted_terms = extract_last_non_empty_line(reasoning_output)
        df.at[idx, 'Terms'] = extracted_terms
        print(f"Processed {idx}: {title}, Terms: {extracted_terms}")

        count_index += 1
        # 保存逻辑
        if (count_index + 1) % 50 == 0:  # 每处理50条保存一次
            df.to_excel('scientific_terms_all_last.xlsx')
            
        # **清理 GPU 内存**
        del inputs, output, prompt, reasoning_output  # 删除变量
        torch.cuda.empty_cache()  # 释放显存
             
    except Exception as e:
        print(f"Error processing {title}: {e}")

# %%
df.to_excel('scientific_terms_all_last.xlsx')




# %% read address
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
count_index = 0
# Iterate over each row in journal_abb
for idx, row in df.iterrows():
    
    title = row['Article Title']
    abstract = row['Abstract']
    reasoning_output = row['Reasoning']
    terms = row['Terms']

    # 如果有
    if len(terms) < 800:
        continue  # 跳过此次迭代
    
    try:
        prompt  = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
#Task: Extract scientifically meaningful key terms from the given academic TITLE and ABSTRACT, following these rules:
# 1. Identify technical terms representing distinct scientific concepts/methods/materials
# 2. Capitalize only the first letter of the first word in the extracted term unless specific academic conventions dictate otherwise (e.g., proper nouns, established formulas, or field-specific norms)
# 3. Follow the common rules for using hyphens for specific terms in academia
# 4. Retain full original terminology if the abbreviation is not very well-known
# 5. Exclude generic prepositions/articles (of/the/for/etc.)
# 6. Prioritize multi-word technical phrases over single words
# 7. Ensure the output terms are concise and avoid repetition or similar vocabulary, outputting each unique term only once

#TITLE:{title}
#ABSTRACT:{abstract}

#Output format: ONLY output extracted terms in plain text (without any bolding, line breaking, or numbering) and DO Not output any unnecessary content. Use semicolons to separate the extracted academic terms"""

        # **Tokenize 并生成输出**
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                           truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保数据在 GPU/CPU 上

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, 
                                    top_k=50, top_p=0.9, temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id)

        # **解码输出**
        reasoning_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # **存入 DataFrame**
        df.at[idx, 'Reasoning'] = reasoning_output
        extracted_terms = extract_last_non_empty_line(reasoning_output)
        df.at[idx, 'Terms'] = extracted_terms
        print(f"Processed {idx}: {title}, Terms: {extracted_terms}")

        count_index += 1
        # 保存逻辑
        if (count_index + 1) % 50 == 0:  # 每处理50条保存一次
            df.to_excel('scientific_terms_all_last.xlsx')
            
        # **清理 GPU 内存**
        del inputs, output, prompt, reasoning_output  # 删除变量
        torch.cuda.empty_cache()  # 释放显存
             
    except Exception as e:
        print(f"Error processing {title}: {e}")

# %%
df.to_excel('scientific_terms_all_last.xlsx')


# %% read address
df = pd.read_excel('scientific_terms_all_last.xlsx')
df

# %%
count_index = 0
# Iterate over each row in journal_abb
for idx, row in df.iterrows():
    
    title = row['Article Title']
    abstract = row['Abstract']
    reasoning_output = row['Reasoning']
    terms = row['Terms']

    # 如果有
    if  bool(re.match(r'^[A-Za-z]', terms)):
        continue  # 跳过此次迭代
    
    try:
        prompt  = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
#Task: Extract scientifically meaningful key terms from the given academic TITLE and ABSTRACT, following these rules:
# 1. Identify technical terms representing distinct scientific concepts/methods/materials
# 2. Capitalize only the first letter of the first word in the extracted term unless specific academic conventions dictate otherwise (e.g., proper nouns, established formulas, or field-specific norms)
# 3. Follow the common rules for using hyphens for specific terms in academia
# 4. Retain full original terminology if the abbreviation is not very well-known
# 5. Exclude generic prepositions/articles (of/the/for/etc.)
# 6. Prioritize multi-word technical phrases over single words
# 7. Ensure the output terms are concise and avoid repetition or similar vocabulary, outputting each unique term only once

#TITLE:{title}
#ABSTRACT:{abstract}

#Output format: ONLY output extracted terms in plain text (without any bolding, line breaking, or numbering) and DO Not output any unnecessary content. Use semicolons to separate the extracted academic terms"""

        # **Tokenize 并生成输出**
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, 
                           truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 确保数据在 GPU/CPU 上

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=4096, do_sample=True, 
                                    top_k=50, top_p=0.9, temperature=0.7,
                                    pad_token_id=tokenizer.pad_token_id)

        # **解码输出**
        reasoning_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # **存入 DataFrame**
        df.at[idx, 'Reasoning'] = reasoning_output
        extracted_terms = extract_last_non_empty_line(reasoning_output)
        df.at[idx, 'Terms'] = extracted_terms
        print(f"Processed {idx}: {title}, Terms: {extracted_terms}")

        count_index += 1
        # 保存逻辑
        if (count_index + 1) % 50 == 0:  # 每处理50条保存一次
            df.to_excel('scientific_terms_all_last.xlsx')
            
        # **清理 GPU 内存**
        del inputs, output, prompt, reasoning_output  # 删除变量
        torch.cuda.empty_cache()  # 释放显存
             
    except Exception as e:
        print(f"Error processing {title}: {e}")

# %%
df.to_excel('scientific_terms_all_last.xlsx')
