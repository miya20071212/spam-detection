import pandas as pd
import re

# 读取原始数据
with open('带标签短信.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

labels = []
messages = []
for line in lines:
    line = line.strip()
    if line:
        labels.append(line[0])          # 标签
        messages.append(line[1:].strip())  # 短信内容

df = pd.DataFrame({'label': labels, 'message': messages})

# ---------- 清洗 ----------
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['message'].apply(clean_text)
df = df[df['cleaned'] != '']

# 只保留标签和清洗后的文本
df = df[['label', 'cleaned']]

# ---------- 保存 ----------
df.to_csv('clean_data.csv', index=False, encoding='utf-8-sig')
print(f"清洗完成，共 {len(df)} 条数据，已保存 clean_data.csv")