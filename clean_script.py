# clean_script.py
# 垃圾短信与诈骗邮件智能检测 - 数据清洗脚本（批量处理版）

import pandas as pd
from ultraclean.clean import cleanup

# ========== 1. 读取原始数据 ==========
# 假设原始文件为 SMSSpamCollection，tab分隔，无表头
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

print("数据集大小：", df.shape)
print("原始标签分布：\n", df['label'].value_counts())

# ========== 2. 批量清洗 ==========
df['cleaned_message'] = df['message'].apply(lambda text: cleanup(text))

# ========== 3. 保存清洗结果 ==========
# 只保留标签和清洗后的文本
clean_df = df[['label', 'cleaned_message']]
clean_df.to_csv('raw_clean.csv', index=False)

print("清洗完成，已保存 raw_clean.csv")