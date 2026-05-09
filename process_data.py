import pandas as pd
import re

# 读取原始数据 (格式: 每行第一个字符是标签0/1，后面紧跟短信内容)
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
print("原始数据量:", len(df))

# ---------- 中文清洗 ----------
def clean_text(text):
    # 去掉网址
    text = re.sub(r'https?://\S+', '', text)
    # 只保留中英文、数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned'] = df['message'].apply(clean_text)
df = df[df['cleaned'] != '']  # 删除清洗后为空的行

# ---------- 1:1 下采样 ----------
spam = df[df['label'] == '1']   # 垃圾短信
ham  = df[df['label'] == '0']   # 正常短信

n = min(len(spam), len(ham))
spam_down = spam.sample(n=n, random_state=42)
ham_down  = ham.sample(n=n, random_state=42)

balanced = pd.concat([spam_down, ham_down]).sample(frac=1, random_state=42).reset_index(drop=True)

print("平衡后正负样本数:")
print(balanced['label'].value_counts())

# ---------- 保存最终文件 ----------
balanced.to_csv('clean_data.csv', index=False, encoding='utf-8-sig')
print("已保存 clean_data.csv，可直接用于模型训练")