# %%
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import subprocess
import pandas as pd
import numpy as np
import os
from transformers import AutoConfig
from tqdm import tqdm
import re

def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    memory_info = result.stdout.decode('utf-8').strip().split(',')
    total_memory, free_memory, used_memory = map(int, memory_info)
    return total_memory, free_memory, used_memory

total_memory, free_memory, used_memory = get_gpu_memory_usage()
print(f"Total Memory: {total_memory} MiB")
print(f"Free Memory: {free_memory} MiB")
print(f"Used Memory: {used_memory} MiB")

#model_path = "../backup/pretrainedmodel/llama-3.1-8b-instruct"
model_path = r"D:\PreTrainedModels\Llama3-KALE-LM-Chem-1.5-8B"

# %%
df = pd.read_excel('term_counts_cluster.xlsx')  # 20万词
terms = df['Term'].tolist()  # 术语列表
print(f"Length of terms: {len(terms)}")
df

# %%
accelerator = Accelerator()

config = AutoConfig.from_pretrained(model_path)
max_length = config.max_position_embeddings
print(f"Max input length for the model is: {max_length}")

# %%
# load a base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map="auto",
    torch_dtype=torch.bfloat16, # default is float32
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   
tokenizer.pad_token = tokenizer.eos_token
# 在加载tokenizer后添加以下配置
tokenizer.padding_side = 'left'  # 关键修复

# %%
# 准备prompt模板
def create_prompt(term):
    return f"""Please determine if this term is related to Artificial Intelligence (AI). 
Consider models, architectures, algorithms, methods, evaluation metrics, etc.
Respond ONLY with 'True' or 'False' without any punctuation or explanations.
Term: {term}
Answer:"""

# 修改后的生成参数
generation_config = {
    "max_new_tokens": 5,          # 限制生成的最大token数
    "do_sample": False,           # 禁用采样，使用贪婪解码
    "temperature": 1.0,           # 设置为默认值（即使不使用采样）
    "top_p": 1.0,                 # 禁用top-p采样
    "repetition_penalty": 1.1,    # 轻微惩罚重复
    "length_penalty": 1.0,        # 不额外惩罚生成长度
    "num_beams": 1,               # 禁用beam search（单束搜索）
    "pad_token_id": tokenizer.eos_token_id,  # 显式设置pad token
    "eos_token_id": tokenizer.eos_token_id   # 显式设置结束token
}

# 生成prompt列表
prompts = [create_prompt(term) for term in terms]

# 批处理参数设置
batch_size = 32  # 根据GPU内存调整
results = []
max_new_tokens = 5

# 使用Accelerator准备组件
model = accelerator.prepare(model)

# 带进度条的批处理
with tqdm(total=len(prompts), desc="Processing terms") as pbar:
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # 编码输入
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True  # 确保特殊token正确添加
        ).to(model.device)
        
        # 生成响应
        outputs = model.generate(
            **inputs,
            **generation_config,  # 使用更新后的参数
        )   
        
        # 解码并提取结果
        input_lengths = inputs.attention_mask.sum(dim=1)
        batch_results = []
        for idx, output in enumerate(outputs):
            # 提取生成部分
            generated = output[input_lengths[idx]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            
            # 使用正则表达式匹配结果
            match = re.search(r'(true|false)', answer, re.IGNORECASE)
            if match:
                batch_results.append(match.group(0).lower() == 'true')
            else:
                batch_results.append(False)  # 无法识别时默认False
        
        results.extend(batch_results)
        pbar.update(len(batch_prompts))
        
# 将结果保存到DataFrame
df['AI-related term'] = results

# %%
df.to_excel('term_counts_cluster_class.xlsx',index=False)

# %%