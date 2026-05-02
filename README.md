# spam-detection
3# 垃圾短信与诈骗邮件的智能检测

## 文件说明
- `SMSSpamCollection`：原始短信数据集（UCI）
- `clean_data.csv`：最终平衡后的数据集，可用于训练模型
- `clean_script.py`：数据清洗脚本，读取原始数据并输出清洗后的 raw_clean.csv
- `downsample.py`：下采样脚本，读取 raw_clean.csv，进行正负样本 1:1 平衡，输出 clean_data.csv
- `raw_clean.csv`：清洗后的中间数据（含标签和清洗后短信）
