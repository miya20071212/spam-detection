import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("🕒 阶段 1：加载数据与切分...")
start_time = time.time()

# 1. 读取数据
df = pd.read_csv('processed_data_with_tokens.csv')

# 2. 清理可能的空值 (保证安全)
df = df.dropna(subset=['tokenized_message', 'label'])

# 3. 提取特征(X)和标签(y)
X_raw = df['tokenized_message']
y = df['label']  # 正常是0，垃圾是1

# 4. 切分数据集 (80%训练，20%测试，固定 random_state=42)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)
print(f"✅ 数据切分完毕！训练集: {len(X_train_raw)}条，测试集: {len(X_test_raw)}条.")


print("\n🕒 阶段 2：提取 TF-IDF 特征 (5000维)...")
# 限制 5000 维，和基线模型保持完全一致
tfidf = TfidfVectorizer(max_features=5000)

# 注意：训练集用 fit_transform，测试集用 transform
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)
print(f"✅ 特征提取完毕！特征矩阵形状: {X_train_tfidf.shape}")


print("\n🕒 阶段 3 & 4：构建 LinearSVC 并开始网格搜索 (GridSearch)...")
search_start = time.time()

# 定义基础模型
svm_model = LinearSVC(random_state=42, max_iter=2000, dual=False)

# 定义要搜索的参数网格 (惩罚系数 C)
# C越小越防止过拟合，C越大越拟合训练集
param_grid = {
    'C': [0.1, 1.0, 10.0]
}

# 配置网格搜索
# cv=5 表示5折交叉验证，n_jobs=4 表示使用你的 Ultra 7 的 4个核心并行计算（防止16G内存爆掉）
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro', # 因为垃圾短信检测，F1分数比单纯的Accuracy更重要
    n_jobs=4, 
    verbose=2 # 打印训练过程
)

# 开始暴力搜索最强参数！
grid_search.fit(X_train_tfidf, y_train)

search_end = time.time()
print(f"\n🎉 网格搜索完成！耗时: {search_end - search_start:.2f} 秒")
print(f"🏆 搜索到的最佳参数组合: {grid_search.best_params_}")

# ==========================================
# 阶段 5：在测试集上进行最终评估
# ==========================================
print("\n🕒 阶段 5：在测试集上评估最佳模型...")
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_tfidf)

print("\n📊 === SVM 测试集最终成绩单 ===")
print(f"Accuracy (准确率): {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告 (Precision, Recall, F1):")
print(classification_report(y_test, y_pred, digits=4))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

print(f"\n✅ 全部流程运行完毕！总耗时: {time.time() - start_time:.2f} 秒")