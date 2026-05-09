# spam-detection
3# 垃圾短信与诈骗邮件的智能检测

## 文件说明
Part-A：
- `clean_data.csv`：最终平衡后的数据集，可用于训练模型
- `process_data.py`：数据清洗脚本，进行正负样本 1:1 平衡
Part-B：分词与TF-IDF提取。生成词云图、数据分布图
- `B.py`：代码实现
- `label_pie_chart.png`：展示样本是否平衡（饼图）
- `length_distribution.png`：展示垃圾短信与正常短信特征差异（长度分布图）
- `spam_wordcloud.png`：垃圾短信的关键词特征（词云图）
- `processed_data_with_tokens.rar`：传递给C的关键文件。解压为processed_data_with_tokens.csv，组员C对tokenized_message列（已分词）做TF-IDF（建议设为5000维），再接贝叶斯分类器即可。
