import pandas as pd

# 读取清洗后的数据
df = pd.read_csv('raw_clean.csv')

print("下采样前标签分布：")
print(df['label'].value_counts())

# 分离垃圾（spam）和正常（ham）
spam = df[df['label'] == 'spam']
ham  = df[df['label'] == 'ham']

# 取较小类的数量
n_min = min(len(spam), len(ham))

# 下采样（随机抽取，保持可重复性）
spam_down = spam.sample(n=n_min, random_state=42)
ham_down  = ham.sample(n=n_min, random_state=42)

# 合并、打乱顺序
balanced = pd.concat([spam_down, ham_down]).sample(frac=1, random_state=42).reset_index(drop=True)

print("平衡后标签分布：")
print(balanced['label'].value_counts())

# 输出最终版
balanced.to_csv('clean_data.csv', index=False)
print("已保存 clean_data.csv")