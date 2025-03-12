# %%
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# 设置字体路径，这里以Windows为例，其他操作系统路径可能不同
font_path = "C:/Windows/Fonts/arial.ttf"  # Windows的Arial字体路径

# %%
df = pd.read_excel('term_counts_cluster_class_edit.xlsx')
df

# %%
df_ai = df[df['AI-related term'] == True]
df_ai

# %%
# 设置随机种子值，确保每次生成相同的词云
random_state = 42  # 固定种子

# 将DataFrame转换为字典形式，key是term，value是count
terms_dict = df_ai.set_index('Term')['Count'].to_dict()

# 创建WordCloud对象并自定义配置
wordcloud = WordCloud(width=2500, height=900,  # 设置词云图的尺寸
                      min_font_size=10, max_font_size=140,  # 设置最小和最大的字体大小
                      background_color='white',
                      font_path=font_path,
                      random_state=random_state).generate_from_frequencies(terms_dict)

# 显示词云图
plt.figure(figsize=(25, 9))  # 可以在这里调整最终图像的尺寸
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 不显示坐标轴
# 保存图像，指定dpi为600
plt.savefig("wordcloud_ai.png", dpi=600, bbox_inches='tight', format='png')

# %%
# %%
df_non_ai = df[df['AI-related term'] == False]
df_non_ai

# %%
# 将DataFrame转换为字典形式，key是term，value是count
terms_dict = df_non_ai.set_index('Term')['Count'].to_dict()

# 创建WordCloud对象并自定义配置
wordcloud = WordCloud(width=2500, height=900,  # 设置词云图的尺寸
                      min_font_size=10, max_font_size=80,  # 设置最小和最大的字体大小
                      background_color='white',
                      font_path=font_path,
                      random_state=random_state).generate_from_frequencies(terms_dict)

# 显示词云图
plt.figure(figsize=(25, 9))  # 可以在这里调整最终图像的尺寸
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 不显示坐标轴
# 保存图像，指定dpi为600
plt.savefig("wordcloud_non_ai.png", dpi=600, bbox_inches='tight', format='png')

# %%
