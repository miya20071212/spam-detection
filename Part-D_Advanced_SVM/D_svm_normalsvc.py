import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def write_log(text):
    """将输出同时打印到终端和日志文件，防止半夜终端清空"""
    print(text)
    with open("D_svm_gridsearch_log.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")

write_log("🌙 夜间炼丹模式启动！(满血 5 折交叉验证版)")
start_time = time.time()

# ==========================================
# 1. 数据加载与预处理
# ==========================================
write_log("\n🕒 阶段 1：加载数据与切分...")
df = pd.read_csv('../data/processed_data_with_tokens.csv')
df = df.dropna(subset=['tokenized_message', 'label'])

X_raw = df['tokenized_message']
y = df['label']  

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)
write_log(f"✅ 数据切分完毕！训练集: {len(X_train_raw)}条，测试集: {len(X_test_raw)}条.")

# ==========================================
# 2. 特征提取 (TF-IDF)
# ==========================================
write_log("\n🕒 阶段 2：提取 TF-IDF 特征 (5000维)...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_raw)
X_test_tfidf = tfidf.transform(X_test_raw)
write_log(f"✅ 特征提取完毕！特征矩阵形状: {X_train_tfidf.shape}")

# ==========================================
# 3. 网格搜索 (GridSearchCV)
# ==========================================
write_log("\n🕒 阶段 3：构建普通 SVC 并开始网格搜索...")
write_log("☕ 采用 cv=5，共需进行 15 次独立训练。预计耗时 1.5 ~ 2.5 小时。")
write_log("主帅请去睡觉，剩下的交给我！")

base_svm = SVC(random_state=42)

param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1.0, 10.0]
}

# cv 修改为 5，n_jobs 保持 2 以防止 16GB 内存溢出
grid_search = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=2, 
    verbose=3 
)

search_start = time.time()
grid_search.fit(X_train_tfidf, y_train)
search_end = time.time()

write_log(f"\n🎉 太阳升起了！网格搜索圆满完成！耗时: {(search_end - search_start)/60:.2f} 分钟")
write_log(f"🏆 搜索到的最佳参数组合: {grid_search.best_params_}")

# ==========================================
# 4. 模型评估
# ==========================================
write_log("\n🕒 阶段 4：在测试集上评估最佳模型...")
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_tfidf)

write_log("\n📊 === 普通 SVC (cv=5最优版) 成绩单 ===")
write_log(f"Accuracy (准确率): {accuracy_score(y_test, y_pred):.4f}")
write_log("\n分类报告:")
write_log(classification_report(y_test, y_pred, digits=4))
write_log("\n混淆矩阵:")
write_log(str(confusion_matrix(y_test, y_pred)))

# ==========================================
# 5. 模型保存
# ==========================================
write_log("\n💾 阶段 5：正在保存最终模型...")
joblib.dump(best_svm, 'best_standard_svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer_std.pkl')
write_log("✅ 模型保存成功！晚安，主帅！")

write_log(f"\n🏁 全部流程运行完毕！总耗时: {(time.time() - start_time)/60:.2f} 分钟")