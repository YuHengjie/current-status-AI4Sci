# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %% read addresses
data = pd.read_excel('address_final.xlsx',index_col=0)
data

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
data = data[data['Source Title'].isin(target_titles)].reset_index(drop=True)
data

# %%
publication_count = data.groupby('Publication Year').size().reset_index(name='AI publications')
publication_count

# %%
# 示例数据
x = np.array(range(10))  # X轴为0到9
y = publication_count['AI publications'].values

# 定义指数函数形式
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# 使用curve_fit来拟合指数函数
params, covariance = curve_fit(exp_func, x, y)

# 打印拟合参数
print(f'拟合参数: a={params[0]}, b={params[1]}, c={params[2]}')

# 绘制原始数据和拟合曲线
plt.scatter(x, y, label='Original data')
plt.plot(x, exp_func(x, *params), 'r-', label='Fitted curve', linewidth=2,color='#FF8C00')
plt.title('Exponential Regression for AI Publications')
plt.xlabel('Index (0-9)')
plt.ylabel('Number of AI Publications')
plt.legend()
plt.show()

# %%
# 预测值
y_pred = exp_func(x, *params)

# 计算R^2
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f'R-squared: {r_squared}')

# %%
plt.figure(figsize=(4.8, 1.3)) # 设置画布尺寸
bars = plt.bar(
publication_count.index.astype(str), # 确保索引转换为字符串格式
publication_count['AI publications'],
color='gray', # 使用Matplotlib默认蓝色
width=0.7 # 调整柱子宽度
)

plt.ylabel('Total count')

# 获取当前坐标轴
ax = plt.gca()
# 隐藏上、右边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置刻度只在底部和左侧显示
ax.tick_params(
    axis='both',         # 同时设置x和y轴
    which='both',        # 同时设置主刻度和小刻度
    bottom=True,         # 保留底部刻度
    left=True,           # 保留左侧刻度
    top=False,           # 隐藏顶部刻度
    right=False          # 隐藏右侧刻度
)

# 设置x轴刻度为2015-2024
plt.xticks(ticks=publication_count.index.astype(str), labels=publication_count['Publication Year'])

plt.tight_layout()
plt.savefig('ai_publication_part_count_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_data = np.array(range(10))  # 原始数据的x值

# 定义指数函数形式
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# 使用curve_fit来拟合指数函数
params, covariance = curve_fit(exp_func, x_data, publication_count['AI publications'])

# 创建一组更密集的x值用于画出平滑曲线
x_dense = np.linspace(min(x_data), max(x_data), 300)  # 生成300个点以确保曲线平滑
y_dense_pred = exp_func(x_dense, *params)

# 创建图形并设置尺寸
plt.figure(figsize=(4.8, 1.3))

# 绘制柱状图
bars = plt.bar(
    publication_count.index.astype(str), # 确保索引转换为字符串格式
    publication_count['AI publications'],
    color='gray',           # 使用灰色填充柱子
    width=0.7               # 调整柱子宽度
)

# 在同一图表上绘制拟合曲线
plt.plot(x_dense, y_dense_pred, 'r-', label='Fitted curve', linewidth=2,color='#FF8C00')

plt.ylabel('Total count')

# 获取当前坐标轴
ax = plt.gca()
# 隐藏上、右边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# 设置x轴刻度为2015-2024
plt.xticks(ticks=publication_count.index.astype(str), labels=publication_count['Publication Year'])

# 设置刻度只在底部和左侧显示
ax.tick_params(
    axis='both',         # 同时设置x和y轴
    which='both',        # 同时设置主刻度和小刻度
    bottom=True,         # 保留底部刻度
    left=True,           # 保留左侧刻度
    top=False,           # 隐藏顶部刻度
    right=False          # 隐藏右侧刻度
)

plt.tight_layout()
plt.savefig('ai_publication_part_count_bar_with_smooth_curve.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
