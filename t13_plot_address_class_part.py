# %%
import matplotlib.pyplot as plt
import pandas as pd
from brokenaxes import brokenaxes
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %%
df = pd.read_excel('address_part_year_count_mean_ci_95.xlsx',index_col=0)
df

# %%
plt.figure(figsize=(5, 3.5))  # 在此处定义图形大小

# 创建断裂轴对象（调整参数获得最佳显示效果）
bax = brokenaxes(
    ylims=((0, 1), (3.5, 10)),  # 定义要放大的两个Y轴区间
    hspace=0.1,  # 两个子图之间的间距
    despine=False,  # 保留轴线
    
)

# 定义指标配置（保持原有颜色方案）
metrics = [
    {
        'mean_col': 'address number_mean',
        'lower_col': 'address number_lower_ci',
        'upper_col': 'address number_upper_ci',
        'label': 'All addresses',
        'color': '#09c892'
    },
    {
        'mean_col': 'Science_institution_count_mean',
        'lower_col': 'Science_institution_count_lower_ci',
        'upper_col': 'Science_institution_count_upper_ci',
        'label': 'Science institution',
        'color': '#6d66bd'
    },
    {
        'mean_col': 'AI_institution_count_mean',
        'lower_col': 'AI_institution_count_lower_ci',
        'upper_col': 'AI_institution_count_upper_ci',
        'label': 'AI institution',
        'color': '#ec6a0d'
    }
]

# 绘制所有指标
for metric in metrics:
    # 绘制折线（自动适应断裂轴）
    bax.plot(
        df['Publication Year'], 
        df[metric['mean_col']],
        color=metric['color'],
        label=metric['label'],
        linewidth=1.5
    )
    
    # 绘制置信区间（自动截断）
    bax.fill_between(
        df['Publication Year'],
        df[metric['lower_col']],
        df[metric['upper_col']],
        color=metric['color'],
        alpha=0.4,
        edgecolor=None,  # 或直接省略这个参数
        linewidth=0      # 确保无描边线
    )

bax.set_xticks(range(2015,2025))

# 设置标签和标题
bax.set_ylabel('Mean count', fontsize=10, labelpad=30)

# 调整图例位置
bax.legend(loc='upper right',  frameon=False) # bbox_to_anchor=(0.5, 1),


plt.savefig('address_part_year_count_mean_ci_95.png', dpi=300, bbox_inches='tight')
#plt.tight_layout()
plt.show()



# %%
df = pd.read_excel('address_part_first_year_count_mean_ci_95.xlsx',index_col=0)
df

# %%
plt.figure(figsize=(5, 3.5))  # 定义图形大小

# 创建普通坐标轴
ax = plt.gca()

# 定义指标配置（保持原有颜色方案）
metrics = [
    {
        'mean_col': 'first_Science_position_mean',
        'lower_col': 'first_Science_position_lower_ci',
        'upper_col': 'first_Science_position_upper_ci',
        'label': 'Science institution',
        'color': '#6d66bd'
    },
    {
        'mean_col': 'first_AI_position_mean',
        'lower_col': 'first_AI_position_lower_ci',
        'upper_col': 'first_AI_position_upper_ci',
        'label': 'AI institution',
        'color': '#ec6a0d'
    },
]

# 绘制所有指标
for metric in metrics:
    # 绘制折线
    ax.plot(
        df['Publication Year'], 
        df[metric['mean_col']],
        color=metric['color'],
        label=metric['label'],
        linewidth=1.5
    )
    
    # 绘制置信区间
    ax.fill_between(
        df['Publication Year'],
        df[metric['lower_col']],
        df[metric['upper_col']],
        color=metric['color'],
        alpha=0.4,
        edgecolor=None,  # 或直接省略这个参数
        linewidth=0      # 确保无描边线
    )

# 设置坐标轴参数
ax.set_xticks(range(2015, 2025))
ax.set_ylim(0, 6)  # 设置完整的Y轴范围

# 设置标签
ax.set_ylabel('First address rank', fontsize=10)

# 调整图例位置
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig('address_part_first_year_count_mean_ci_95.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
df = pd.read_excel('address_part_AI_sum_year.xlsx')
df

# %%
# 数据示例（替换成你的df）
years = df['Publication Year']
counts = df['sum_has_AI_institution']
percentages = df['Percentage']

# 创建画布和主Y轴
fig, ax1 = plt.subplots(figsize=(5, 3.5))

# 绘制柱状图（主Y轴）
bars = ax1.bar(years, counts, color='#45c3af', alpha=0.7, label='Count')
ax1.set_ylabel('Article count with AI institutions', )
ax1.tick_params(axis='y', )

ax1.set_xticks(ticks=range(2015, 2025))        # 设置刻度位置
ax1.set_xticklabels(labels=range(2015, 2025))  # 设置刻度标签

# 创建次Y轴
ax2 = ax1.twinx()

# 绘制折线图（次Y轴）
# 绘制带空心标记的折线
line = ax2.plot(years, percentages, 
                marker='o', 
                linestyle='--', 
                color='#021e30', 
                markersize=6,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor='#021e30',
                label='Percentage')

ax2.set_ylabel('Percentage in all articles (%)',)
ax2.tick_params(axis='y', )

ax2.set_xticks(ticks=range(2015, 2025))        # 设置刻度位置
ax2.set_xticklabels(labels=range(2015, 2025))  # 设置刻度标签

# 添加图例
lines = [bars, line[0]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=False)

# 自动调整布局
fig.tight_layout()
plt.savefig('address_part_AI_sum_year.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
df = pd.read_excel('address_part_first_AI_sum_year.xlsx')
df

# %%
# 数据示例（替换成你的df）
years = df['Publication Year']
counts = df['sum_first_AI_position_1']
percentages = df['Percentage']

# 创建画布和主Y轴
fig, ax1 = plt.subplots(figsize=(5, 3.5))

# 绘制柱状图（主Y轴）
bars = ax1.bar(years, counts, color='#45c3af', alpha=0.7, label='Count')
ax1.set_ylabel('Article count with AI institution as first affiliation', )
ax1.tick_params(axis='y', )

ax1.set_xticks(ticks=range(2015, 2025))        # 设置刻度位置
ax1.set_xticklabels(labels=range(2015, 2025))  # 设置刻度标签

# 创建次Y轴
ax2 = ax1.twinx()

# 绘制折线图（次Y轴）
# 绘制带空心标记的折线
line = ax2.plot(years, percentages, 
                marker='o', 
                linestyle='--', 
                color='#021e30', 
                markersize=6,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor='#021e30',
                label='Percentage')

ax2.set_ylabel('Percentage in all articles (%)',)
ax2.tick_params(axis='y', )

ax2.set_xticks(ticks=range(2015, 2025))        # 设置刻度位置
ax2.set_xticklabels(labels=range(2015, 2025))  # 设置刻度标签

# 添加图例
lines = [bars, line[0]]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=False)

# 自动调整布局
fig.tight_layout()
plt.savefig('address_part_first_AI_sum_year.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
