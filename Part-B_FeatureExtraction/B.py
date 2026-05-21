import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# ==========================================
# 1. 环境配置与中文支持
# ==========================================
# 解决绘图时中文显示乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 读取数据
# ==========================================
print("正在读取数据...")
# 告诉程序用 GBK 编码（专门处理中文 Windows 文件的编码）来读取
df = pd.read_csv('../data/clean_data.csv', encoding='GBK')
# ==========================================
# 3. 文本预处理（分词）
# ==========================================
def chinese_tokenizer(text):
    # 确保文本是字符串，并使用jieba分词
    return " ".join(jieba.cut(str(text)))

print("正在进行文本分词...")
df['tokenized_message'] = df['cleaned_message'].apply(chinese_tokenizer)

# ==========================================
# 4. 特征提取 (TF-IDF)
# ==========================================
print("正在提取 TF-IDF 特征...")
# 提取前5000个关键词特征
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['tokenized_message'])
print(f"TF-IDF 特征提取完成，特征矩阵形状: {tfidf_matrix.shape}")

# ==========================================
# 5. 数据可视化 (至少3种图表)
# ==========================================

# --- 图表 1：标签分布饼图 ---
print("正在生成饼图...")
plt.figure(figsize=(8, 6))
label_counts = df['label'].value_counts()
plt.pie(label_counts, labels=['正常 (Ham)', '垃圾 (Spam)'],
        autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999'])
plt.title('短信类型分布比例 (证明数据已平衡)')
plt.savefig('label_pie_chart.png')

# --- 图表 2：短信长度分布图 ---
print("正在生成长度分布图...")

# 【修复点】在这里补上计算长度的代码，确保 x 轴有数据
df['msg_len'] = df['cleaned_message'].astype(str).apply(len)

# 映射标签
label_map = {0: '正常 (Ham)', 1: '垃圾 (Spam)', '0': '正常 (Ham)', '1': '垃圾 (Spam)'}
df['label_display'] = df['label'].map(label_map)

plt.figure(figsize=(10, 6))
# 绘制图表
sns.histplot(data=df, x='msg_len', hue='label_display', element='step', kde=True, palette='magma')

plt.title('正常与垃圾短信句子长度对比')
plt.xlabel('字符长度')
plt.ylabel('频数')
# 删掉 plt.legend 那行，sns.histplot 会自动根据 hue 生成带中文的图例
plt.xlim(0, 300)
plt.savefig('length_distribution.png')

# --- 图表 3：高频词云图 ---
print("正在生成词云图...")
spam_text = " ".join(df[df['label'] == 1]['tokenized_message'])
# 注意：如果运行报错说找不到 simhei.ttf，请确保路径正确或更换字体
wordcloud = WordCloud(
    font_path='simhei.ttf',
    width=800,
    height=400,
    background_color='white'
).generate(spam_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('垃圾短信高频关键词词云')
plt.savefig('spam_wordcloud.png')

# ==========================================
# 6. 保存结果供第三人使用
# ==========================================
# 保存带有分词结果的CSV，方便队友直接跑模型
df.to_csv('processed_data_with_tokens.csv', index=False)
print("\n任务全部完成！")
print("已生成文件：")
print("1. processed_data_with_tokens.csv (数据特征)")
print("2. label_pie_chart.png (饼图)")
print("3. length_distribution.png (长度分布图)")
print("4. spam_wordcloud.png (词云图)")