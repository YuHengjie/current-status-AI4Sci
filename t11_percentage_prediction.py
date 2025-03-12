# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics  import r2_score
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'

# %%
data = pd.read_excel('publication_sum.xlsx',index_col=0)
data

# %%
plt.figure(figsize=(4.8, 1.3)) # 设置画布尺寸
bars = plt.bar(
data.index.astype(str), # 确保索引转换为字符串格式
data['Percent'],
color='gray', # 使用Matplotlib默认蓝色
width=0.7 # 调整柱子宽度
)

plt.ylim([0,4])

plt.ylabel('Percentage')

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

plt.tight_layout()
plt.savefig('ai_publication_percent_bar.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# %%
# %%
# 示例数据
x = np.array(range(10))  # X轴为0到9
y = data['Percent'].values

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

# %% 拟合及预测
# 历史数据 (2015-2024)
years_historical = np.array([2015,  2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
percentage_historical =  data['Percent'].values
 
# 未来估计数据 (示例：2025-2030)
years_future = np.array([2025, 2028, 2030, 2033, 2035, 2040, 2050])
percentage_future_estimate = np.array([4.5,  7.7, 10.5, 15, 18, 22.7, 25])  # 用户需替换为实际数据 
 
# 合并数据集（若未来数据可信度高）
all_years = np.concatenate((years_historical,  years_future))
all_percentage = np.concatenate((percentage_historical,  percentage_future_estimate))
 
# ====== 模型定义 ======
def logistic_model(t, K, r, t0):
    """Logistic增长模型 
    t: 时间（年份）
    K: 饱和值（最大占比%）
    r: 增长率 
    t0: 拐点对应年份 
    """
    return K / (1 + np.exp(-r  * (t - t0)))
 
# ====== 参数拟合 ======
# 时间归一化（以起始年2015为t=0）
t_data = all_years - 2015  
y_data = all_percentage 
 
# 初始参数猜测（参考历史趋势：K≈20, r≈0.3, t0≈20对应2035年）
initial_guess = [25, 0.3, 15]  
 
# 执行拟合（bounds限制参数物理意义：K>0, r>0）
params, covariance = curve_fit(logistic_model, t_data, y_data, p0=initial_guess, 
                              bounds=([0.1, 0.1, 0], [50, 2, 50]))
 
K_fit, r_fit, t0_fit = params 
fit_errors = np.sqrt(np.diag(covariance))   # 参数标准差 
 
# ====== 模型评估 ======
y_pred = logistic_model(t_data, K_fit, r_fit, t0_fit)
r_squared = r2_score(y_data, y_pred)
 
# ====== 可视化 ======
plt.figure(figsize=(10,  6))
years_extended = np.arange(2015,  2051)  # 延伸至2030年 
t_extended = years_extended - 2015 
y_fit_extended = logistic_model(t_extended, K_fit, r_fit, t0_fit)
 
# 绘制历史数据与未来估计 
plt.scatter(years_historical,  percentage_historical, c='blue', label='Historical Data')
plt.scatter(years_future,  percentage_future_estimate, c='orange', marker='x', label='Future Estimates')
 
# 绘制拟合曲线 
plt.plot(years_extended,  y_fit_extended, 'r--', label=f'Logistic Fit (R²={r_squared:.3f})')
 
plt.xlabel('Year') 
plt.ylabel('AI  Paper Percentage (%)')
plt.title('Logistic  Model Fitting for AI Research Adoption')
plt.legend() 
plt.grid(True) 
#plt.savefig('percentage_prediction.png',dpi=300)
plt.show() 
 
# ====== 输出结果 ======
print(f"【拟合参数】")
print(f"饱和值 K = {K_fit:.2f} ± {fit_errors[0]:.2f}%")
print(f"增长率 r = {r_fit:.3f} ± {fit_errors[1]:.3f}/year")
print(f"拐点年份 t0 = {2015 + t0_fit:.1f} ± {fit_errors[2]:.1f} year")
print(f"拟合优度 R² = {r_squared:.4f}")

# %% 绘制最终预测
K = 25
r = 0.240
t0 = 16.5  # 拐点对应归一化时间（实际年份：2015+16.5≈2031.5）

# ====== 数据准备 ======
# 历史数据（2015-2024）
years_historical = np.array([2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]) 
percentage_historical = data['Percent'].values

# 生成预测时间轴（2015-2050）
years_full = np.arange(2015,  2051)
t_full = years_full - 2015  # 时间归一化

# 分割线型区间
split_year = 2024
mask_historical = years_full <= split_year
mask_forecast = years_full >= split_year

# ====== Visualization ======
plt.figure(figsize=(8, 3))
 
# 绘制预测曲线（分实线/虚线）
plt.plot(years_full[mask_historical],  logistic_model(t_full[mask_historical], K, r, t0),
         color='#23BCC0', linewidth=2, label='Historical fit')
plt.plot(years_full[mask_forecast],  logistic_model(t_full[mask_forecast], K, r, t0),
         color='#F67970', linestyle='--', linewidth=2, label='Future forecast')
 
# 添加真实数据散点 
plt.scatter(years_historical,  percentage_historical,
           color='#23BCC0', 
           edgecolor='white',  # 关键修改点 
           linewidths=0.5,
           zorder=10,
           s=40, 
           label='Observed data')

# ====== 预测区间计算 ======
# 提取预测段数据（2025-2050）
mask_forecast = years_full > split_year

years_forecast = years_full[mask_forecast]
forecast_values = logistic_model(t_full[mask_forecast], K, r, t0)
 
# 计算20%波动带（工程容差法）
lower_bound = forecast_values * 0.8  # 下界 
upper_bound = forecast_values * 1.2  # 上界 
 
# ====== 可视化增强 ======
# 添加预测区间带 
plt.fill_between(years_forecast,  lower_bound, upper_bound,
                 color='#F67970', alpha=0.15,  # 主色透明度适配 
                 edgecolor='none', zorder=5,
                 label='Uncertainty band')

# 提取2024年观测值（假设为数据集最新值）
obs_2024 = percentage_historical[years_historical == 2024][0]  
 
# 计算2025年预测基准值 
pred_2025 = logistic_model(2025-2015, K, r, t0)  # 25.1%（归一化t=10）
 
# 构建过渡区间坐标轴 
years_transition = np.array([2024,  2025])  # 过渡时间轴 
lower_trans = np.array([obs_2024,  pred_2025*0.8])  # 下界序列 
upper_trans = np.array([obs_2024,  pred_2025*1.2])  # 上界序列 
 
# 添加过渡区填充 
plt.fill_between(years_transition,  lower_trans, upper_trans,
                 color='#F67970', alpha=0.15, zorder=5,
                 edgecolor='none', )

# 获取图例句柄与标签（正确API调用）
handles, labels = plt.gca().get_legend_handles_labels() 
 
# 定义图例层级顺序（工业可视化标准）
legend_order = {
    'Observed data': 0,      # 观测数据（最高优先级）
    'Historical fit': 1,      # 历史拟合 
    'Future forecast': 2,     # 预测曲线 
    'Uncertainty band':3  # 置信区间 
}
 
# 按自定义顺序重排 
sorted_pairs = sorted(zip(handles, labels), 
                     key=lambda x: legend_order.get(x[1],  999))
handles_sorted, labels_sorted = zip(*sorted_pairs)
 
# 应用增强型图例 
plt.legend(handles_sorted,  labels_sorted,
           loc='upper left', 
           bbox_to_anchor=(0.02, 0.98),
           ncol=1, 
           frameon=False,
           fontsize=9)

plt.ylim(-0.5,  30.5)  # 强制y轴范围 
#plt.xlabel("Year",  fontsize=11)
plt.ylabel("Percentage (%)", fontsize=11)
plt.xticks(np.arange(2015,  2051, 5))

plt.savefig('percentage_prediction.tiff',dpi=600)

# %%
