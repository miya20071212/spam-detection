# spam-detection
## 垃圾短信与诈骗邮件的智能检测

### 文件说明

Part-A:  完成数据清洗与采样平衡脚本，并提供平衡后的数据集
- `clean_data.csv`：最终平衡后的数据集，可用于训练模型  
- `process_data.py`：数据清洗脚本，进行正负样本 1:1 平衡  

Part-B: 分词与TF-IDF提取。生成词云图、数据分布图  
- `B.py`：代码实现  
- `label_pie_chart.png`：展示样本是否平衡（饼图）  
- `length_distribution.png`：展示垃圾短信与正常短信特征差异（长度分布图）  
- `spam_wordcloud.png`：垃圾短信的关键词特征（词云图）  
- `processed_data_with_tokens.rar`：传递给C的关键文件。解压为 `processed_data_with_tokens.csv`，组员C对 `tokenized_message` 列（已分词）做 TF-IDF（建议设为5000维），再接贝叶斯分类器即可。  

Part-C: 基线模型构建与测试集初步评估  
- `C_nb.py`：第9周基线模型代码，使用 TF-IDF + 朴素贝叶斯分类器  
- `C_result.txt`：第9周首次跑通基线模型后的结果记录  
- `C_week10_eval.py`：第10周评估代码，对基线模型进行稳定运行检查，并补充测试集评估指标  
- `week10_baseline_report.txt`：第10周测试集评估结果，包括 Accuracy、Precision、Recall、F1-score、ROC-AUC、混淆矩阵等  
- `week10_predictions.csv`：测试集预测结果  
- `week10_misclassified.csv`：测试集中误分类样本，便于后续误差分析和报告撰写

Part-D：进阶模型构建与评估
- `D_svm_linearsvc.py`：LinearSVM模型代码，使用TF-IDF+线性分类器（线性核）
- `D_svm_linearsvc_result.txt`：LinearSVM模型结果记录
- `D_svm_normalsvc.py`：正常SVM模型代码，使用TF-IDF+非线性分类器（RBF核）
- `D_svm_normalsvc_result.txt`：正常SVM模型结果记录

### 当前进度说明

Part-A 已完成：  
- 完成原始数据清洗  
- 完成正负样本 1:1 平衡  
- 输出最终训练数据 `clean_data.csv`  

Part-B 已完成：  
- 完成文本分词处理  
- 完成词云图、样本分布图、长度分布图  
- 输出 `processed_data_with_tokens.csv` 供后续模型训练使用  

Part-C 当前已完成：  
- 完成 TF-IDF（5000维）+ 朴素贝叶斯 基线模型  
- 完成测试集初步评估  
- 完成预测结果与误分类样本导出  
- 连续运行两次结果一致，说明基线模型代码可以稳定运行，结果可复现  

Part-D 当前已完成：
- 完成基于 TF-IDF（5000维）+ LinearSVC 的进阶模型构建与评估
- 完成基于 TF-IDF（5000维）+ 普通SVC 的进阶模型构建与评估

### MultinomialNB模型结果

使用数据：`processed_data_with_tokens.csv`  
使用列：`tokenized_message`  
模型：TF-IDF（5000维） + MultinomialNB  

测试结果：  
- Accuracy：0.9645  
- Precision：0.9487  
- Recall：0.9821  
- F1-score：0.9651  
- Macro F1：0.9645  
- ROC-AUC：0.9954

混淆矩阵：  
```text
[[15151  849]
 [  286 15714]]
```

### SVM-LinearSVC模型结果

使用数据：`processed_data_with_tokens.csv`  
使用列：`tokenized_message`  
模型：TF-IDF（5000维） + LinearSVC  

测试结果：  
- Accuracy：0.9842  
- Precision：0.9874  
- Recall：0.9810  
- F1-score：0.9842  
- Macro F1：0.9842    

混淆矩阵：  
```text
[[15760   201]
 [  304 15735]]
```

### SVM-普通SVC模型结果

使用数据：`processed_data_with_tokens.csv`  
使用列：`tokenized_message`  
模型：TF-IDF（5000维） + 普通SVC  

测试结果：  
- Accuracy：0.9863  
- Precision：0.9904  
- Recall：0.9823  
- F1-score：0.9863  
- Macro F1：0.9863    

混淆矩阵：  
```text
[[15760   153]
 [  284 15735]]
```

