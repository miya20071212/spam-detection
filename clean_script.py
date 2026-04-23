# clean_script.py
# 垃圾短信与诈骗邮件智能检测 - 数据清洗脚本

from ultraclean.clean import cleanup
from ultraclean.predict import Spam

def clean_and_detect(text):
    """清洗文本并检测是否为垃圾信息"""
    # 清洗：去除网址、邮箱、特殊符号等
    cleaned = cleanup(text)
    
    # 检测是否为垃圾信息
    detector = Spam()
    is_spam = detector.predict(cleaned)
    
    return cleaned, is_spam

# 测试样例
test_messages = [
    "Congratulations! You've won a FREE trip! Click here to claim: https://scam.com",
    "Hi Bob, are we still meeting for lunch tomorrow?"
]

print("=== 垃圾短信与诈骗邮件检测结果 ===\\n")
for msg in test_messages:
    clean_text, spam_flag = clean_and_detect(msg)
    print(f"原始: {msg}")
    print(f"清洗后: {clean_text}")
    print(f"结果: {'是垃圾信息' if spam_flag else '正常信息'}\\n")