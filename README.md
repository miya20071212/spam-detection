# spam-detection

# 垃圾短信与诈骗邮件的智能检测

## 项目简介

本项目旨在通过自然语言处理（NLP）与机器学习/深度学习技术，对海量短信与诈骗邮件进行智能分类，实现垃圾信息的自动检测与精准拦截。

项目从数据清洗、文本特征提取逐步演进至传统机器学习模型与深度学习模型构建，目前已经实现：

- TF-IDF + MultinomialNB 基线模型
- TF-IDF + SVM 进阶模型
- 基于 PyTorch 的 LSTM 深度学习模型

并对不同模型进行系统化性能对比分析，最终构建高精度垃圾短信检测系统。

---

# 项目目录结构

为了保证代码的规范性与可维护性，本项目采用模块化工程结构，公共数据统一存放，各阶段代码与产出物相互隔离。

```text
spam-detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── clean_data.csv
│   ├── processed_data_with_tokens.csv
│   └── *.rar
│
├── Part-A_DataCleaning/
│   └── process_data.py
│
├── Part-B_FeatureExtraction/
│   ├── B.py
│   ├── label_pie_chart.png
│   ├── length_distribution.png
│   └── spam_wordcloud.png
│
├── Part-C_Baseline_NB/
│   ├── C_nb.py
│   ├── C_week10_eval.py
│   ├── week10_baseline_report.txt
│   ├── week10_predictions.csv
│   └── week10_misclassified.csv
│
├── Part-D_Advanced_SVM/
│   ├── D_svm_linearsvc.py
│   ├── D_svm_normalsvc.py
│   ├── D_svm_linearsvc_result.txt
│   ├── D_svm_normalsvc_result.txt
│   └── best_standard_svm_model.pkl
│
└── Part-E_DeepLearning_LSTM/
    ├── E_lstm_model.py
    ├── E_lstm_result.txt
    ├── lstm_training_log.txt
    └── best_lstm_model.pth
```

---

# 项目环境依赖

## Python 环境

推荐版本：

```bash
Python 3.10+
```

---

## requirements.txt

项目根目录已提供：

```text
requirements.txt
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## requirements.txt 内容

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jieba>=0.42.1
wordcloud>=1.9.2
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
joblib>=1.3.0
tqdm>=4.65.0
```

---

# 各模块说明

## 0. data/（公共数据依赖）

集中存放各阶段生成的数据文件，供后续模型统一调用。

### 文件说明

| 文件名 | 说明 |
|---|---|
| clean_data.csv | A阶段输出的平衡数据集（1:1） |
| processed_data_with_tokens.csv | B阶段输出的标准分词数据 |
| *.rar | 原始/处理后数据压缩包 |

---

## 1. Part-A_DataCleaning（数据清洗与类别平衡）

### 核心文件

| 文件名 | 说明 |
|---|---|
| process_data.py | 数据清洗、去噪、样本平衡脚本 |

### 功能

- 删除网址
- 删除特殊字符
- 删除空文本
- 完成正负样本 1:1 平衡

---

## 2. Part-B_FeatureExtraction（特征工程与可视化）

### 核心文件

| 文件名 | 说明 |
|---|---|
| B.py | 分词与数据可视化主程序 |

### 可视化结果

| 文件名 | 说明 |
|---|---|
| label_pie_chart.png | 类别分布饼图 |
| length_distribution.png | 文本长度分布图 |
| spam_wordcloud.png | 垃圾短信高频词词云 |

### 功能

- 中文分词
- 词频统计
- 文本长度分析
- 数据可视化
- 输出标准化分词数据

---

## 3. Part-C_Baseline_NB（朴素贝叶斯基线模型）

### 核心文件

| 文件名 | 说明 |
|---|---|
| C_nb.py | TF-IDF + MultinomialNB 基线模型 |
| C_week10_eval.py | 第10周稳定性评估脚本 |

### 输出结果

| 文件名 | 说明 |
|---|---|
| week10_baseline_report.txt | 模型评估报告 |
| week10_predictions.csv | 测试集预测结果 |
| week10_misclassified.csv | 误分类样本 |

### 功能

- TF-IDF 向量化
- MultinomialNB 训练
- 分类性能评估
- 混淆矩阵生成

---

## 4. Part-D_Advanced_SVM（SVM 进阶模型）

### 核心文件

| 文件名 | 说明 |
|---|---|
| D_svm_linearsvc.py | LinearSVC 模型 |
| D_svm_normalsvc.py | RBF核 SVC + GridSearch |

### 输出结果

| 文件名 | 说明 |
|---|---|
| D_svm_linearsvc_result.txt | LinearSVC 结果 |
| D_svm_normalsvc_result.txt | RBF核SVC结果 |
| best_standard_svm_model.pkl | 最优SVM模型权重 |

### 功能

- LinearSVC 分类
- RBF 核 SVC 分类
- GridSearch 参数调优
- 模型保存与加载

---

## 5. Part-E_DeepLearning_LSTM（LSTM 深度学习模型）

### 核心文件

| 文件名 | 说明 |
|---|---|
| E_lstm_model.py | LSTM 网络训练脚本 |

### 输出结果

| 文件名 | 说明 |
|---|---|
| E_lstm_result.txt | LSTM 模型评估结果 |
| lstm_training_log.txt | 训练日志 |
| best_lstm_model.pth | 最优 LSTM 模型权重 |

### 功能

- 构建词表
- Embedding 词嵌入
- LSTM 序列建模
- Early Stopping 防止过拟合
- 自动保存最佳模型

---

# 当前项目进度

| 模块 | 状态 | 内容 |
|---|---|---|
| Part-A | 已完成 | 数据清洗与类别平衡 |
| Part-B | 已完成 | 分词与可视化 |
| Part-C | 已完成 | 朴素贝叶斯基线模型 |
| Part-D | 已完成 | SVM 模型优化 |
| Part-E | 已完成 | LSTM 深度学习模型 |

---

# 数据集说明

项目原始数据总量约 80 万条：

| 类别 | 数量 |
|---|---|
| 正常短信 | 719945 |
| 垃圾短信 | 约80000 |

由于原始数据存在类别不平衡问题，项目采用：

```text
1:1 下采样策略
```

构建平衡数据集：

| 类别 | 数量 |
|---|---|
| 正常短信 | 80000 |
| 垃圾短信 | 80000 |

最终训练数据总量：

```text
160000 条
```

---

# 核心模型性能对比

## 1. MultinomialNB 基线模型

### 模型

```text
TF-IDF（5000维） + MultinomialNB
```

### 性能指标

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9645 |
| Precision | 0.9487 |
| Recall | 0.9821 |
| F1-score | 0.9651 |

### 混淆矩阵

```text
[[15151   849]
 [  286 15714]]
```

---

## 2. LinearSVC 模型

### 模型

```text
TF-IDF（5000维） + LinearSVC
```

### 性能指标

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9842 |
| Precision | 0.9874 |
| Recall | 0.9810 |
| F1-score | 0.9842 |

### 混淆矩阵

```text
[[15760   201]
 [  304 15735]]
```

---

## 3. RBF 核 SVC 模型

### 模型

```text
TF-IDF（5000维） + SVC(RBF)
```

### 性能指标

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9863 |
| Precision | 0.9904 |
| Recall | 0.9823 |
| F1-score | 0.9863 |

### 混淆矩阵

```text
[[15760   153]
 [  284 15735]]
```

---

## 4. LSTM 深度学习模型（当前最佳）

### 模型

```text
Embedding + LSTM
```

### 性能指标

| 指标 | 数值 |
|---|---|
| Accuracy | 0.9888 |
| Precision | 0.9934 |
| Recall | 0.9841 |
| F1-score | 0.9887 |

### 混淆矩阵

```text
[[15856   105]
 [  255 15784]]
```

---

# 项目阶段总结

随着模型不断演进，系统整体性能得到显著提升。

相比于初期朴素贝叶斯模型，LSTM 神经网络不仅取得当前最高综合准确率（98.88%），同时显著降低了正常短信误判率（False Positives），将误杀数量从 849 条降低至 105 条，在保证高召回率的同时进一步提升了实际业务应用价值。

后续项目将继续探索更高性能的深度学习模型与文本表示方法，以进一步提升垃圾短信与诈骗邮件检测系统的泛化能力与鲁棒性。
