# %%
import pandas as pd
import numpy as np

# %% read addresses
address_df = pd.read_excel('address_final.xlsx',index_col=0)
address_df

# %%
# 假设 address_df 是你的 DataFrame
# 定义需要筛选的特定值列表
target_titles = [
    "NATURE",
    "SCIENCE",
    "SCIENCE ADVANCES",
    "NATURE COMMUNICATIONS",
    "PROCEEDINGS OF THE NATIONAL ACADEMY OF SCIENCES OF THE UNITED STATES OF AMERICA"
]

# 筛选 Source Title 列中包含特定值的行
address_df = address_df[address_df['Source Title'].isin(target_titles)].reset_index(drop=True)
address_df

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
result.to_excel('address_part_year_count_mean_std_100.xlsx')
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
result.to_excel('address_part_year_count_mean_std_97.xlsx')
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
final_result.to_excel('address_part_year_count_mean_ci_95.xlsx')
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
final_result_filtered.to_excel('address_part_first_year_count_mean_ci_95.xlsx')
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

final_df.to_excel("address_part_first_AI_sum_year.xlsx", index=False)
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

final_df.to_excel("address_part_AI_sum_year.xlsx", index=False)
final_df

# %%
