import pandas as pd
import re

# 1. 读取原始数据
with open('带标签短信.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

labels = []
messages = []
for line in lines:
    line = line.strip()
    if line:
        labels.append(line[0])
        messages.append(line[1:].strip())

df = pd.DataFrame({'label': labels, 'message': messages})

# 2. 标签转整数
df['label'] = df['label'].astype(int)

# 3. 清洗
def clean_text(text):
    if type(text) != str:
        return ''
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['message'].apply(clean_text)
df = df[df['cleaned'] != '']

# 4. 删除原始短信列
df = df[['label', 'cleaned']]

# 5. 1:1 下采样
spam = df[df['label'] == 1]
ham  = df[df['label'] == 0]

n = min(len(spam), len(ham))
spam_down = spam.sample(n=n, random_state=42)
ham_down  = ham.sample(n=n, random_state=42)

balanced = pd.concat([spam_down, ham_down])
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. 保存
balanced.to_csv('clean_data.csv', index=False, encoding='utf-8-sig')