# %%
import pandas as pd

# %%
# 读取Excel文件
df = pd.read_excel('article_combine.xlsx')

# 数据预处理
# 确保列名正确（根据实际列名调整）
df = df.rename(columns={
    'Source Title': 'Journal',
    'Publication Year': 'Year',
    'Article Title': 'Title'
})

# 清洗数据：去除空值，转换年份格式
df = df.dropna(subset=['Journal', 'Year'])
df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')[0]  # 提取4位数字年份
df = df[df['Year'].str.isnumeric()]  # 过滤无效年份
df['Year'] = df['Year'].astype(int)

# 筛选2015-2024年的数据（根据需求调整范围）
df = df[df['Year'].between(2015, 2024)]
df

# %%
# 聚合统计
result = (
    df.groupby(['Journal', 'Year'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# 补全所有年份列（确保2015-2024列都存在）
all_years = list(range(2015, 2025))
for year in all_years:
    if year not in result.columns:
        result[year] = 0

# 整理列顺序
result = result[['Journal'] + all_years]

# 重命名年份列
result.columns = ['Journal'] + [str(y) for y in all_years]

# 按期刊名称排序
result = result.sort_values('Journal').reset_index(drop=True)
result

# %%
# 新增：添加汇总行 (这里开始新增部分)
# 计算各列总和（排除Journal列）
sum_row = result.iloc[:, 1:].sum().to_dict()
sum_row['Journal'] = 'Total'  # 设置汇总行标识

# 创建汇总行DataFrame并与原数据合并
sum_df = pd.DataFrame([sum_row])
result = pd.concat([result, sum_df], ignore_index=True)
result

# %%
result.to_excel('ai_publication_year.xlsx', index=False)

# %%
