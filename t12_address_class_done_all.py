# %%
import pandas as pd
import numpy as np

# %% read addresses
data_df = pd.read_excel('article_combine.xlsx')
address_df = data_df[['DOI', 'Publication Year', 'Source Title', 'Article Title', 'Addresses']]
address_df = address_df[
    (address_df['Publication Year'] >= 2015) & 
    (address_df['Publication Year'] <= 2024)
]
address_df

# %%
# Remove text within square brackets and the brackets themselves
address_df['Addresses_edit'] = address_df['Addresses'].str.replace(r'\[.*?\]', '', regex=True)

# Replace multiple spaces with a single space and strip leading/trailing spaces
address_df['Addresses_edit'] = address_df['Addresses_edit'].str.replace(r'\s+', ' ', regex=True).str.strip()
address_df

# %%
# 删除 Addresses_edit 为空的行（包括 NaN 和空字符串）
address_df = address_df[
    address_df["Addresses_edit"].notna() & 
    (address_df["Addresses_edit"].str.strip() != "")
]

# 可选：重置索引（避免删除后的索引不连续）
address_df = address_df.reset_index(drop=True)
address_df

# %%
# Define a function to calculate the address number
def calculate_address_number(addresses):
    if pd.isna(addresses):
        return 1  # or any default value you prefer for NaN cases
    addresses_list = addresses.split('; ')
    # Filter out any empty strings resulting from split
    addresses_list = [addr for addr in addresses_list if addr.strip()]
    return len(addresses_list) + 1

# Apply the function to create the new column
address_df['address number'] = address_df['Addresses_edit'].apply(calculate_address_number)
address_df

# %%
top_10_values = address_df['address number'].nlargest(500).tolist()
print("最大的十个数:", top_10_values)

# %%
address_class_df = pd.read_excel('address_classification_all.xlsx',index_col=0)
address_class_df

# %%
# 创建地址到分类的映射字典，标准化地址（去除首尾空格）
class_dict = pd.Series(
    address_class_df['classification'].values,
    index=address_class_df['address'].str.strip()
).to_dict()
class_dict

# %%
# 处理Addresses_edit，生成对应的分类列
address_df['address classes'] = address_df['Addresses_edit'].apply(
    lambda x: '; '.join(
        [class_dict.get(addr.strip(), 'Unclear institution') 
         for addr in x.split(';')]
    )
)

address_df

# %%
# 新增判断列：存在 "AI institution" 则为 1，否则为 0
address_df["has_AI_institution"] = (
    address_df["address classes"]
    .str.split("; ")           # 按分号拆分为列表
    .apply(                    # 检查列表中是否包含目标分类
        lambda lst: 1 if "AI institution" in lst else 0
    )
)
print(address_df["has_AI_institution"].sum())
address_df

# %%
# 新增列：标记第一个 "AI institution" 的位置（从1开始计数）
address_df["first_AI_position"] = address_df["address classes"].apply(
    lambda x: next(
        (i + 1 for i, cls in enumerate(x.split("; ")) if cls == "AI institution"),
        0  # 默认值（未找到时返回0）
    )
)
address_df

# %%
# 新增列：标记第一个 "AI institution" 的位置（从1开始计数）
address_df["first_Science_position"] = address_df["address classes"].apply(
    lambda x: next(
        (i + 1 for i, cls in enumerate(x.split("; ")) if cls == "Science institution"),
        0  # 默认值（未找到时返回0）
    )
)
address_df

# %%
# 新增列：提取第一个分类，处理空值及异常情况
address_df["first_class"] = (
    address_df["address classes"]
    .str.split("; ")              # 按分隔符拆分字符串为列表
    .apply(lambda x: x[0] if len(x) > 0 else "No Class")  # 取第一个元素，空列表返回"No Class"
    .fillna("Invalid Data")       # 处理原始值为 NaN 的情况
)
address_df

# %%
# 新增列：统计 "AI institution" 出现的次数
address_df["AI_institution_count"] = (
    address_df["address classes"]
    .fillna("")  # 将 NaN 替换为空字符串，避免后续操作报错
    .apply(
        lambda x: x.split("; ").count("AI institution")
    )
)
address_df

# %%
# 新增列：统计 "AI institution" 出现的次数
address_df["Science_institution_count"] = (
    address_df["address classes"]
    .fillna("")  # 将 NaN 替换为空字符串，避免后续操作报错
    .apply(
        lambda x: x.split("; ").count("Science institution")
    )
)
address_df

# %%
address_df.to_excel('address_final.xlsx')

# %%
# 按年份分组并计算统计量
result = address_df.groupby("Publication Year").agg({
    "address number": ["mean", "std"],
    "AI_institution_count": ["mean", "std"],
    "Science_institution_count": ["mean", "std"]
})

# 重命名列（可选，提升可读性）
result.columns = [
    "address_mean", "address_std",
    "AI_count_mean", "AI_count_std",
    "Science_count_mean", "Science_count_std"
]

# 重置索引（将年份从索引变为普通列）
result = result.reset_index()
result.to_excel('address_year_count_mean_std_100.xlsx')
result

# %%
def truncate_top_3_percent(group):
    stats = {}
    
    # 处理每个列的函数
    def process_column(column, prefix):
        data = group[column].dropna()
        n = len(data)
        if n == 0:
            stats[f'{prefix}_mean'] = np.nan
            stats[f'{prefix}_std'] = np.nan
            return
        
        # 计算排除数量（最多排除n-1个，避免全删）
        exclude_num = min(int(np.ceil(n * 0.03)), n - 1)
        if exclude_num > 0:
            # 降序排序后排除前exclude_num个
            truncated = data.sort_values(ascending=False).iloc[exclude_num:]
        else:
            truncated = data
        
        stats[f'{prefix}_mean'] = truncated.mean()
        stats[f'{prefix}_std'] = truncated.std(ddof=1)  # 与原std()一致
    
    # 对各列应用处理
    process_column('address number', 'address')
    process_column('AI_institution_count', 'AI_count')
    process_column('Science_institution_count', 'Science_count')
    
    return pd.Series(stats)

# 应用分组处理
result = address_df.groupby('Publication Year').apply(truncate_top_3_percent).reset_index()

# 列名与原始结构保持一致
result.columns = [
    'Publication Year',
    'address_mean', 'address_std',
    'AI_count_mean', 'AI_count_std',
    'Science_count_mean', 'Science_count_std'
]
result.to_excel('address_year_count_mean_std_97.xlsx')
result


# %%
from scipy import stats
# 定义计算置信区间的函数
def calculate_ci(group, confidence=0.95):
    n = len(group)
    mean = np.mean(group)
    std = np.std(group, ddof=1)  # 无偏标准差（分母 n-1）
    
    # 选择 Z 值或 t 值（根据样本量）
    if n >= 30:
        z = stats.norm.ppf(1 - (1 - confidence)/2)  # Z 值（大样本）
        margin = z * std / np.sqrt(n)
    else:
        t = stats.t.ppf(1 - (1 - confidence)/2, df=n-1)  # t 值（小样本）
        margin = t * std / np.sqrt(n)
    
    return pd.Series({
        "mean": mean,
        "lower_ci": mean - margin,
        "upper_ci": mean + margin
    })
    
# 定义目标列
target_columns = ["address number", "AI_institution_count", "Science_institution_count"]

# 按年份分组，逐列计算置信区间
results = {}
for col in target_columns:
    ci_df = address_df.groupby("Publication Year")[col].apply(calculate_ci).unstack()
    ci_df.columns = [f"{col}_mean", f"{col}_lower_ci", f"{col}_upper_ci"]
    results[col] = ci_df

# 合并所有结果
final_result = pd.concat(results.values(), axis=1).reset_index()
final_result.to_excel('address_year_count_mean_ci_95.xlsx')
final_result

# %%
# 定义目标列（需先去除0值后计算）
target_columns_filtered = ["first_AI_position", "first_Science_position"]

results_filtered = {}
for col in target_columns_filtered:
    # 去除当前列中值为0的行
    filtered_df = address_df[address_df[col] != 0]
    # 按年份分组，逐列计算置信区间
    ci_df = filtered_df.groupby("Publication Year")[col].apply(calculate_ci).unstack()
    ci_df.columns = [f"{col}_mean", f"{col}_lower_ci", f"{col}_upper_ci"]
    results_filtered[col] = ci_df

# 合并所有结果
final_result_filtered = pd.concat(results_filtered.values(), axis=1).reset_index()
final_result_filtered.to_excel('address_first_year_count_mean_ci_95.xlsx')
final_result_filtered

# %%
# 过滤出 first_AI_position == 1 的行，按年份分组求和
ai_first_pos_sum = (
    address_df[address_df["first_AI_position"] == 1]  # 筛选值为1的行
    .groupby("Publication Year")                      # 按年份分组
    ["first_AI_position"].sum()                       # 对目标列求和
    .reset_index()                                    # 将年份从索引转为列
    .rename(columns={"first_AI_position": "sum_first_AI_position_1"})  # 重命名结果列
)
# 新增步骤：计算全量数据的年度计数
yearly_counts = address_df.groupby("Publication Year").size().reset_index(name='total_publications')

# 合并两个结果
final_df = pd.merge(
    ai_first_pos_sum,
    yearly_counts,
    on="Publication Year",
    how="left"  # 保留所有有AI=1记录的年份
)

final_df['Percentage'] = final_df['sum_first_AI_position_1']/final_df['total_publications']*100

final_df.to_excel("address_first_AI_sum_year.xlsx", index=False)
final_df

# %%
# 过滤出 has_AI_institution == 1 的行，按年份分组求和
ai_first_pos_sum = (
    address_df[address_df["has_AI_institution"] == 1]  # 筛选值为1的行
    .groupby("Publication Year")                      # 按年份分组
    ["has_AI_institution"].sum()                       # 对目标列求和
    .reset_index()                                    # 将年份从索引转为列
    .rename(columns={"has_AI_institution": "sum_has_AI_institution"})  # 重命名结果列
)
# 新增步骤：计算全量数据的年度计数
yearly_counts = address_df.groupby("Publication Year").size().reset_index(name='total_publications')

# 合并两个结果
final_df = pd.merge(
    ai_first_pos_sum,
    yearly_counts,
    on="Publication Year",
    how="left"  # 保留所有有AI=1记录的年份
)

final_df['Percentage'] = final_df['sum_has_AI_institution']/final_df['total_publications']*100

final_df.to_excel("address_AI_sum_year.xlsx", index=False)
final_df

# %%
