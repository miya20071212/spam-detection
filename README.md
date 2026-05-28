# spam-detection

# 垃圾短信与诈骗邮件的智能检测

## 项目简介

本项目基于自然语言处理（NLP）、机器学习与深度学习技术，对海量短信与诈骗邮件进行智能分类，实现垃圾信息自动检测与精准拦截。

项目从数据清洗与特征工程开始，逐步构建：

- TF-IDF + MultinomialNB 基线模型
- TF-IDF + SVM 进阶模型
- 基于 PyTorch 的 LSTM 深度学习模型

并对不同模型进行系统化性能对比分析，最终实现高精度垃圾短信检测系统。

项目采用模块化工程结构，涵盖：

- 数据预处理
- 中文分词
- 数据可视化
- 特征工程
- 机器学习模型
- 深度学习模型
- 模型评估
- 模型保存与推理

具备较好的工程实践价值与可扩展性。

---

# 一、项目环境依赖

## 1. Python 环境

推荐环境：

bash Python 3.10+ 

---

## 2. requirements.txt

项目根目录已提供：

text requirements.txt 

安装依赖：

bash pip install -r requirements.txt 

---

## 3. 项目主要依赖库

| 库名 | 作用 |
|---|---|
| pandas | 数据处理 |
| numpy | 数值计算 |
| scikit-learn | 机器学习模型与评估 |
| jieba | 中文分词 |
| matplotlib | 数据可视化 |
| seaborn | 统计图绘制 |
| wordcloud | 词云生成 |
| torch | PyTorch 深度学习框架 |
| torchvision | PyTorch视觉组件 |
| torchaudio | PyTorch音频组件 |
| joblib | 模型保存与加载 |
| tqdm | 训练进度显示 |

---

# 二、requirements.txt 内容

txt pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 jieba>=0.42.1 wordcloud>=1.9.2 torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 joblib>=1.3.0 tqdm>=4.65.0 

---

# 三、项目运行方式

## 1. 数据清洗与类别平衡

进入目录：

bash cd Part-A_DataCleaning 

运行：

bash python process_data.py 

功能：

- 清洗原始短信数据
- 删除网址与特殊字符
- 删除空文本
- 完成正负样本 1:1 平衡

输出：

text data/clean_data.csv 

---

## 2. 文本分词与特征工程

进入目录：

bash cd Part-B_FeatureExtraction 

运行：

bash python B.py 

功能：

- 中文分词
- 文本统计分析
- 文本长度分析
- 词云生成
- 数据可视化

输出：

text data/processed_data_with_tokens.csv 

---

## 3. 朴素贝叶斯基线模型

进入目录：

bash cd Part-C_Baseline_NB 

运行：

bash python C_nb.py 

或：

bash python C_week10_eval.py 

功能：

- TF-IDF 向量化
- MultinomialNB 模型训练
- 测试集评估
- 输出分类报告与混淆矩阵

---

## 4. SVM 模型训练与调优

进入目录：

bash cd Part-D_Advanced_SVM 

运行 LinearSVC：

bash python D_svm_linearsvc.py 

运行 RBF 核 SVC：

bash python D_svm_normalsvc.py 

功能：

- LinearSVC 分类
- RBF 核 SVC 分类
- GridSearch 参数调优
- 保存最佳模型

---

## 5. LSTM 深度学习模型

进入目录：

bash cd Part-E_DeepLearning_LSTM 

运行：

bash python E_lstm_model.py 

功能：

- 构建词表
- Embedding 词嵌入
- LSTM 序列建模
- Early Stopping 防止过拟合
- 自动保存最佳模型权重

---

# 四、项目目录结构

text spam-detection/ │ ├── README.md ├── requirements.txt │ ├── data/ │   ├── clean_data.csv │   ├── processed_data_with_tokens.csv │   └── *.rar │ ├── Part-A_DataCleaning/ │   └── process_data.py │ ├── Part-B_FeatureExtraction/ │   ├── B.py │   ├── label_pie_chart.png │   ├── length_distribution.png │   └── spam_wordcloud.png │ ├── Part-C_Baseline_NB/ │   ├── C_nb.py │   ├── C_week10_eval.py │   ├── week10_baseline_report.txt │   ├── week10_predictions.csv │   └── week10_misclassified.csv │ ├── Part-D_Advanced_SVM/ │   ├── D_svm_linearsvc.py │   ├── D_svm_normalsvc.py │   ├── D_svm_linearsvc_result.txt │   ├── D_svm_normalsvc_result.txt │   └── best_standard_svm_model.pkl │ └── Part-E_DeepLearning_LSTM/     ├── E_lstm_model.py     ├── E_lstm_result.txt     ├── lstm_training_log.txt     └── best_lstm_model.pth 

---

# 五、数据集说明

项目原始数据总量约 80 万条：

| 类别 | 数量 |
|---|---|
| 正常短信 | 719945 |
| 垃圾短信 | 约 80000 |

由于原始数据类别分布严重不平衡，项目采用：

text 1:1 下采样策略 

构建平衡数据集：

| 类别 | 数量 |
|---|---|
| 正常短信 | 80000 |
| 垃圾短信 | 80000 |

最终训练数据总量：

text 160000 条 

---

# 六、模型性能对比

## 1. MultinomialNB 基线模型

模型：

text TF-IDF（5000维） + MultinomialNB 

性能指标：

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9645 |
| Precision | 0.9487 |
| Recall | 0.9821 |
| F1-score | 0.9651 |

混淆矩阵：

text [[15151   849]  [  286 15714]] 

---

## 2. LinearSVC 模型

模型：

text TF-IDF（5000维） + LinearSVC 

性能指标：

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9842 |
| Precision | 0.9874 |
| Recall | 0.9810 |
| F1-score | 0.9842 |

混淆矩阵：

text [[15760   201]  [  304 15735]] 

---

## 3. RBF 核 SVC 模型

模型：

text TF-IDF（5000维） + SVC(RBF) 

性能指标：

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9863 |
| Precision | 0.9904 |
| Recall | 0.9823 |
| F1-score | 0.9863 |

混淆矩阵：

text [[15760   153]  [  284 15735]] 

---

## 4. LSTM 深度学习模型（当前最佳）

模型：

text Embedding + LSTM 

性能指标：

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9888 |
| Precision | 0.9934 |
| Recall | 0.9841 |
| F1-score | 0.9887 |

混淆矩阵：

text [[15856   105]  [  255 15784]] 

---

# 七、项目阶段总结

项目从传统机器学习模型逐步演进至深度学习模型，系统整体性能持续提升。

相比于初期朴素贝叶斯模型，LSTM 模型不仅取得当前最高综合准确率（98.88%），同时显著降低了正常短信误判率（False Positives），将误杀数量从 849 条降低至 105 条，在保证高召回率的同时进一步提升了实际业务应用价值。

项目后续将继续探索更高性能的深度学习模型与文本表示方法，以进一步提升垃圾短信与诈骗邮件检测系统的泛化能力与鲁棒性。
