import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


def read_csv_with_fallback(file_path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 文件读取失败，请检查编码格式。")


def main():
    file_path = "processed_data_with_tokens.csv"
    df = read_csv_with_fallback(file_path)

    print("读取到的列名：", list(df.columns))

    # 你这里已经确认过列名了，就直接用固定列名
    text_col = "tokenized_message"
    raw_text_col = "cleaned_message"
    label_col = "label"

    # 保留需要的列
    use_cols = [text_col, label_col]
    if raw_text_col in df.columns:
        use_cols.append(raw_text_col)

    df = df[use_cols].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    X = df[text_col]
    y = df[label_col]

    if raw_text_col in df.columns:
        raw_text = df[raw_text_col]
    else:
        raw_text = df[text_col]

    # 固定随机种子，保证结果可复现
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X, y, raw_text,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF 5000维
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        max_features=5000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    # 保存测试集预测结果
    pred_df = pd.DataFrame({
        "text": raw_test.values,
        "true_label": y_test.values,
        "pred_label": y_pred,
        "prob_spam": y_prob
    })
    pred_df.to_csv("week10_predictions.csv", index=False, encoding="utf-8-sig")

    # 保存误判样本
    wrong_df = pred_df[pred_df["true_label"] != pred_df["pred_label"]].copy()
    wrong_df.to_csv("week10_misclassified.csv", index=False, encoding="utf-8-sig")

    # 保存评估报告
    report_text = (
        "第10周 C任务：基线模型稳定运行检查与测试集初步评估结果\n"
        "模型：TF-IDF(5000) + MultinomialNB\n"
        f"数据总量：{len(df)}\n"
        f"训练集数量：{len(X_train)}\n"
        f"测试集数量：{len(X_test)}\n\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n"
        f"Weighted F1: {weighted_f1:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        "Confusion Matrix:\n"
        f"{cm}\n\n"
        "Classification Report:\n"
        f"{report}\n"
    )

    with open("week10_baseline_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print("已生成文件：")
    print("1. week10_baseline_report.txt")
    print("2. week10_predictions.csv")
    print("3. week10_misclassified.csv")


if __name__ == "__main__":
    main()
