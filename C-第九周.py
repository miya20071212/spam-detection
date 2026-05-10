import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def read_csv_with_fallback(file_path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception:
            continue
    raise ValueError("CSV 文件读取失败，请检查编码格式。")


def find_column(columns, candidates):
    for col in candidates:
        if col in columns:
            return col
    return None


def main():
    file_path = "processed_data_with_tokens.csv"
    df = read_csv_with_fallback(file_path)

    print("读取到的列名：", list(df.columns))

    token_col = find_column(
        df.columns,
        ["tokenized_message", "tokens", "tokenized_text", "cut_text", "segmented_text"]
    )
    label_col = find_column(
        df.columns,
        ["label", "target", "class", "类别", "标签", "is_spam"]
    )

    if token_col is None:
        raise ValueError("没找到分词后的文本列，请检查是不是叫 tokenized_message。")

    if label_col is None:
        raise ValueError("没找到标签列，请检查是不是叫 label。")

    df = df[[token_col, label_col]].dropna().copy()
    df[token_col] = df[token_col].astype(str).str.strip()
    df = df[df[token_col] != ""]

    # 如果标签是字符串，手动映射
    label_map = {
        "spam": 1,
        "ham": 0,
        "垃圾": 1,
        "正常": 0,
        "诈骗": 1,
        "非诈骗": 0
    }

    def convert_label(x):
        if isinstance(x, str):
            x = x.strip()
            if x in label_map:
                return label_map[x]
        return int(x)

    df[label_col] = df[label_col].apply(convert_label)

    X = df[token_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 这里直接对已经分词好的文本做 TF-IDF
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

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    result_text = (
        "第9周 C任务：TF-IDF + 朴素贝叶斯基线模型\n"
        f"数据总量: {len(df)}\n"
        f"训练集数量: {len(X_train)}\n"
        f"测试集数量: {len(X_test)}\n"
        f"TF-IDF 维度: 5000\n"
        f"Accuracy: {acc:.4f}\n\n"
        "分类报告：\n"
        f"{report}"
    )

    print(result_text)

    with open("C_result.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

    print("\n结果已经保存到 C_result.txt")


if __name__ == "__main__":
    main()
