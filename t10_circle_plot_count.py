# %% 数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %%
df = pd.read_excel('ai_publication_year_with_abb.xlsx')
df = df.drop('Journal', axis=1)
years = list(map(str, range(2015, 2025)))  # 10年数据对应10个环

# 计算总和并排序
df['Total'] = df[years].sum(axis=1)
df = df.sort_values(by='Total', ascending=False).reset_index(drop=True)
df = df.drop('Total', axis=1)

df_plot = df.iloc[1:,:]
df_plot

# %% 可视化参数设置
num_sectors = len(df_plot)       # 期刊数量
num_rings = 10                   # 年份环数
ring_thickness = 0.1             # 环厚度
inner_radius = 1.2               # 中心空白区域半径
cmap = plt.cm.get_cmap('RdBu_r')  # 高对比度色板

# 创建角度参数（右侧留出15度空白）
delta_deg = 8  # 空白区域角度
delta_rad = np.deg2rad(delta_deg)
total_angle = 2*np.pi - delta_rad  # 有效分布角度
theta = np.linspace(delta_rad/2, 2*np.pi - delta_rad/2, num_sectors, endpoint=False)
theta_step = total_angle / num_sectors

# 创建图形
plt.figure(figsize=(20, 16))
ax = plt.subplot(111, polar=True)

# 极坐标样式设置
ax.set_theta_offset(np.pi/2)     # 顶部起始
ax.set_theta_direction(-1)       # 顺时针
ax.set_rticks([])                # 隐藏半径刻度
ax.grid(False)
ax.spines['polar'].set_visible(False)
ax.set_ylim(0, inner_radius + num_rings*ring_thickness)  # 限制半径范围

# 颜色标准化（对数处理）
all_values = df_plot[years].values.flatten()
vmin = max(1, np.min(all_values))  # 避免0值
vmax = np.max(all_values)
norm = LogNorm(vmin=vmin, vmax=vmax)

# 绘制环形图
for ring_idx, year in enumerate(years):
    # 计算环的位置
    bottom = inner_radius + ring_idx * ring_thickness
    values = df_plot[year].values
    
    # 生成颜色
    colors = [cmap(norm(v)) for v in values]
    
    # 绘制环
    ax.bar(theta, 
           height=ring_thickness,
           width=theta_step,
           bottom=bottom,
           color=colors,
           edgecolor='white',
           linewidth=0.3,
           align='edge')

# 颜色条设置（精确居中）
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# 获取圆心在画布中的绝对位置
fig = plt.gcf()
ax_pos = ax.get_position()  # 获取极坐标系子图的位置（归一化坐标）

# 计算圆心坐标（基于子图位置）
center_x = ax_pos.x0 + ax_pos.width/2  # 水平中心
center_y = ax_pos.y0 + ax_pos.height/2  # 垂直中心

# 创建颜色条坐标轴参数（宽度15%，高度1%）
cax_width = 0.2
cax_height = 0.015
cax = fig.add_axes([
    center_x - cax_width/2 - 0.021,  # 水平居中
    center_y - cax_height/2 + 0.06, # 垂直居中
    cax_width,
    cax_height
])

# 绘制颜色条
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Article count', fontsize=12, labelpad=12)
cbar.ax.tick_params(labelsize=12)  # 调整数字12以改变字体大小

# 年份标签（右侧居中）
for ring_idx, year in enumerate(years):
    angle = 0  # 极坐标系0度位置（右侧）
    radius = inner_radius + (ring_idx + 0.5)*ring_thickness
    
    # 添加文本时使用极坐标转换
    ax.text(
        np.deg2rad(0),  # 转换为弧度
        radius,
        f'{year}',
        ha='center',    # 水平居中
        va='center',    # 垂直居中
        rotation=0,     # 保持水平
        fontsize=12,
        color='black',
        # fontweight='bold',
        transform=ax.transProjectionAffine + ax.transAxes  # 使用混合坐标系
    )

# 调整空白区域边界线
ax.plot([0, 0], [inner_radius, inner_radius + num_rings*ring_thickness],
        color='white', lw=2, solid_capstyle='round')

# ========== 极坐标样式设置 ==========
ax.set_theta_offset(np.pi/2)     # 顶部起始
ax.set_theta_direction(-1)       # 顺时针
ax.set_rticks([])                # 隐藏径向刻度
ax.grid(False)                   # 关闭网格
ax.spines['polar'].set_visible(False)  # 隐藏极坐标系边框
ax.set_xticks([])                # 移除角度刻度
ax.set_xticklabels([])           # 清除角度标签
ax.set_facecolor('none')         # 透明背景
ax.set_ylim(0, inner_radius + num_rings*ring_thickness)

# 在绘制环形图后添加以下代码

# 期刊标签参数设置
label_radius = inner_radius + num_rings * ring_thickness + 0.05  # 标签基线半径（比最外环大0.1）
label_fontsize = 12
label_color = 'black'

# 获取期刊缩写列表
abbreviations = df_plot['Abb'].values

# 计算标签旋转角度（考虑极坐标系的偏移和方向）
def get_rotation_angle(radian_angle):
    """将极坐标角度转换为文本旋转角度"""
    # 转换为标准数学角度（0度在右侧，逆时针方向）
    math_angle = np.rad2deg(radian_angle) 
    # 根据极坐标设置调整角度（顶部起始+顺时针）
    adjusted_angle = (90 - math_angle) % 360  # 补偿顶部起始
    return adjusted_angle if adjusted_angle <= 180 else adjusted_angle - 360

# 绘制径向标签
for sector_idx in range(num_sectors):
    angle = theta[sector_idx] + theta_step/2  # 扇区中心角度
    rotation = get_rotation_angle(angle)
    
    ax.text(angle,
            label_radius,
            abbreviations[sector_idx],
            rotation=rotation,
            rotation_mode='anchor',  # 以文本锚点为旋转中心
            ha='left',             
            va='center',  # 垂直对齐
            fontsize=label_fontsize,
            color=label_color,
            #fontweight='bold',
            fontfamily='Arial',
            clip_on=False)

# 调整图形边距（根据标签尺寸可能需要微调）
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 保存和显示
plt.tight_layout()
plt.savefig('ring_chart_count.png', dpi=300, bbox_inches='tight')
plt.show()

# %%