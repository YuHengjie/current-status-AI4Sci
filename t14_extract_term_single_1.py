# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import Dataset
import torch
import pandas as pd

# %% read address
df = pd.read_excel('article_combine.xlsx')
df

# %%
df['Reasoning'] = None
df['Terms'] = None
df

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
# 加载模型和分词器
#model_path = r"D:\PreTrainedModels\DeepSeek-R1-Distill-Llama-8B"
model_path = "../pretrained_model/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
# 将模型移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# %%
# 设置批处理大小（根据GPU显存调整）
batch_size = 16
save_interval = 5  # 每处理5个batch保存一次
start_idx = 0
end_idx = 5000
max_length = 8192
max_new_tokens = 4096

# %%
# 获取需要处理的数据子集
subset = df.iloc[start_idx:end_idx]
total_samples = len(subset)
total_samples

# %%
batch_current = 0
for batch_start in range(0, total_samples, batch_size):
    batch_end = batch_start + batch_size
    batch_indices = subset.iloc[batch_start:batch_end].index
    current_batch = df.loc[batch_indices]
    
    try:
        # 生成当前批次的prompts
        prompts = []
        for idx, row in current_batch.iterrows():
            title = row['Article Title'] or ""  # 处理可能的NaN值
            abstract = row['Abstract'] or ""
            
            prompt = f"""#Role: You are a helpful assistant to extract scientifically meaningful key terms from scientific text.
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
            prompts.append(prompt)

        # 批量编码和生成
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 更新DataFrame
        for j, text in enumerate(generated_texts):
            original_index = batch_indices[j]
            terms = extract_last_non_empty_line(text)
            
            df.at[original_index, 'Reasoning'] = text
            df.at[original_index, 'Terms'] = terms
            print(f"Processed {original_index}: {terms}")
        
        batch_current += 1
        
        if batch_current % save_interval == 0:
            df.to_excel('scientific_terms_5000.xlsx', index=False)
            print(f"===== Auto Saved at batch_{batch_current} =====")
        
        # 显存清理
        del inputs, outputs, generated_texts
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
        # 记录失败批次以便后续重试
        with open("failed_batches.txt", "a") as f:
            f.write(f"Batch {batch_start}-{batch_end}: {str(e)}\n")

# %%
# 最终保存
df.to_excel('scientific_terms_5000.xlsx', index=False)

