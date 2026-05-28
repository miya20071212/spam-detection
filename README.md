# spam-detection
# 垃圾短信与诈骗邮件的智能检测
本项目旨在通过自然语言处理（NLP）与机器学习/深度学习技术，对海量短信与邮件进行智能分类，精准拦截垃圾信息。项目从数据清洗、特征提取，一路演进至基线模型构建、SVM 进阶调优，目前实现了基于 LSTM 神经网络的高精度分类。项目仍在持续迭代优化中。

# 📂 项目目录结构与文件说明
为了保证代码的规范性与可维护性，本项目采用了模块化的工程结构，公共数据统一存放，各阶段代码与产出物互相隔离。

# 0. data/ (公共数据依赖)
集中存放各阶段生成的清洗数据与分词数据，供后续所有模型跨文件夹调用。

clean_data.csv：A阶段输出的目前平衡后的数据集（正负样本 1:1），用于提取特征。
processed_data_with_tokens.csv：B阶段输出的核心文件，包含 tokenized_message 列（已分词），是 C、D、E 阶段所有模型的标准输入。
*.rar：因文件体积较大而切分的原始/处理后数据压缩包。
# 1. Part-A_DataCleaning/ (数据清洗与平衡)
process_data.py：数据预处理脚本，负责清洗冗余信息，并进行正负样本的 1:1 欠采样/过采样平衡。
# 2. Part-B_FeatureExtraction/ (特征工程与可视化)
B.py：分词与数据可视化主脚本。
label_pie_chart.png：样本类别平衡度展示（饼图）。
length_distribution.png：垃圾短信与正常短信的文本长度特征差异（分布图）。
spam_wordcloud.png：垃圾短信的高频关键词特征（词云图）。
# 3. Part-C_Baseline_NB/ (朴素贝叶斯基线模型)
C_nb.py / C-第九周.py：第9周基线模型代码，使用 TF-IDF (5000维) + MultinomialNB。
C_week10_eval.py：第10周评估脚本，用于稳定性测试及补充评估指标。
week10_baseline_report.txt / C_result.txt：评估结果，包含各项指标及混淆矩阵。
week10_predictions.csv：测试集的完整预测概率与结果。
week10_misclassified.csv：测试集中的误分类样本，用于后续误差分析。
# 4. Part-D_Advanced_SVM/ (支持向量机进阶模型)
D_svm_linearsvc.py：使用 TF-IDF + 线性分类器（LinearSVC）。
D_svm_normalsvc.py：使用 TF-IDF + 非线性分类器（普通SVC，RBF核）及网格搜索。
D_svm_linearsvc_result.txt / D_svm_normalsvc_result.txt：SVM 模型的详细运行结果与性能评估。
best_standard_svm_model.pkl：保存的目前最佳 SVC (RBF核) 模型权重文件，可随时加载推理。
# 5. Part-E_DeepLearning_LSTM/ (深度学习进阶模型)
E_lstm_model.py：LSTM 网络训练脚本。包含自定义词表构建、标准的 Train/Val/Test 三分法划分，以及防止过拟合的早停（Early Stopping）和自动存档机制。
E_lstm_result.txt：LSTM 模型的目前成绩单。
lstm_training_log.txt：详细记录了 15 轮次训练中 Train Loss 与 Val Loss 的动态变化，精确捕捉过拟合拐点。
best_lstm_model.pth：PyTorch 保存的目前最佳 LSTM 模型权重（自动锁存在验证集 Loss 最低的 Epoch 4 拐点处）。
# 🚀 当前进度说明
Part-A (已完成)：清洗原始数据并完成正负样本 1:1 平衡。
Part-B (已完成)：完成文本分词、统计学可视化，输出标准化数据 processed_data_with_tokens.csv。
Part-C (已完成)：构建 TF-IDF + 朴素贝叶斯基线模型，完成测试集评估与误差样本导出，验证了数据的有效性与代码的稳定性。
Part-D (已完成)：通过引入 LinearSVC 与非线性 SVC 显著提升了模型的召回率与准确率，确立了传统机器学习的性能天花板。
Part-E (已完成)：引入深度学习框架 PyTorch，利用词嵌入（Embedding）与长短期记忆网络（LSTM）捕获文本序列信息，配合早停策略，取得目前最佳业务指标。
# 📊 核心模型性能对比报告
以下是本项目的四代模型在相同测试集上的目前评估成绩单。

# 1. MultinomialNB 基线模型
特征/模型：TF-IDF（5000维） + MultinomialNB
Accuracy (准确率)：0.9645
Precision (精确率)：0.9487
Recall (召回率)：0.9821
F1-score：0.9651
混淆矩阵：
[[15151   849]
 [  286 15714]]
# 2. SVM - LinearSVC 线性模型
特征/模型：TF-IDF（5000维） + LinearSVC
Accuracy (准确率)：0.9842
Precision (精确率)：0.9874
Recall (召回率)：0.9810
F1-score：0.9842
混淆矩阵：
[[15760   201]
 [  304 15735]]
# 3. SVM - 普通 SVC 非线性模型
特征/模型：TF-IDF（5000维） + 普通SVC (RBF核)
Accuracy (准确率)：0.9863
Precision (精确率)：0.9904
Recall (召回率)：0.9823
F1-score：0.9863
混淆矩阵：
[[15760   153]
 [  284 15735]]
# 4. 深度学习 - LSTM 序列模型 (🏆 目前最佳落地表现)
特征/模型：词嵌入 (5000词表, MAX_LEN=50) + LSTM 神经网络
Accuracy (准确率)：0.9888
Precision (精确率)：0.9934
Recall (召回率)：0.9841
F1-score：0.9887
混淆矩阵：
[[15856   105]
 [  255 15784]]
🔥 目前阶段结论： 随着算法的不断演进，LSTM 神经网络不仅取得了目前最高的综合准确率（98.88%），更展现出了极其优异的商业落地潜质。相比于初期的贝叶斯模型（误杀 849 条），LSTM 凭借对上下文语序的理解，将正常短信的误判拦截数（False Positives）极限压缩至仅 105 条，完美平衡了“高召回”与“极低误杀率”的核心业务需求。后续将基于此基础继续探索更优化的模型架构。
