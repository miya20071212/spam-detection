import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# ==========================================
# ⚙️ 阶段 0：配置设备
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 深度学习引擎已启动！当前使用的计算设备: {device}")

# ==========================================
# 🛠️ 阶段 1：数据加载与严格三分法
# ==========================================
print("\n🕒 阶段 1：数据加载与构建词表...")
df = pd.read_csv('../data/processed_data_with_tokens.csv')
df = df.dropna(subset=['tokenized_message', 'label'])

X_raw = df['tokenized_message'].values
y_raw = df['label'].values

MAX_WORDS = 5000
all_words = " ".join(X_raw).split()
word_counts = Counter(all_words)
common_words = [word for word, count in word_counts.most_common(MAX_WORDS)]

word2idx = {word: idx + 2 for idx, word in enumerate(common_words)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

MAX_LEN = 50 

def encode_text(text):
    tokens = text.split()
    seq = [word2idx.get(word, 1) for word in tokens]
    if len(seq) < MAX_LEN:
        seq = seq + [0] * (MAX_LEN - len(seq)) 
    else:
        seq = seq[:MAX_LEN] 
    return seq

print("正在将文本转换为数字矩阵...")
X_encoded = np.array([encode_text(text) for text in X_raw])
y_encoded = np.array(y_raw, dtype=np.float32)

X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 🧠 阶段 2：搭建网络
# ==========================================
class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SpamLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, (hidden, cell) = self.lstm(embedded)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze()

model = SpamLSTM(vocab_size=MAX_WORDS + 2, embed_dim=64, hidden_dim=128).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 🔥 阶段 3：15轮高强度训练与【自动存档+过拟合记录】
# ==========================================
print("\n🕒 阶段 3：启动 15 轮深度训练，正在监控过拟合过程...")
EPOCHS = 15

# 用于记录绘图数据的历史列表
history = {"train_loss": [], "val_loss": []}
best_val_loss = float('inf')
best_epoch = 0

log_file = "lstm_training_log.txt"
with open(log_file, "w", encoding="utf-8") as f:
    f.write("Epoch,Train_Loss,Val_Loss,Status\n")

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # 训练
    model.train() 
    total_train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # 验证
    model.eval() 
    total_val_loss = 0
    with torch.no_grad(): 
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            val_loss = criterion(predictions, batch_y)
            total_val_loss += val_loss.item()
            
    epoch_time = time.time() - start_time
    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    
    # 【自动存档机制】: 只有当验证集损失比历史最低还低时，才保存模型
    status_str = ""
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        status_str = "★ 最佳存档点"
        print(f"🌟 Epoch {epoch+1:02d}/{EPOCHS} | 耗时: {epoch_time:.1f}秒 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {status_str}")
    else:
        # 如果验证集连续上升，说明已经在过拟合了
        status_str = "⚠ 出现过拟合倾向" if avg_val_loss > history["val_loss"][-2] else ""
        print(f"   Epoch {epoch+1:02d}/{EPOCHS} | 耗时: {epoch_time:.1f}秒 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {status_str}")
        
    # 将过程写入本地日志，方便后续用 Excel 或者是 Matplotlib 画图放入报告
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{status_str}\n")

print(f"\n🎉 15 轮炼丹结束！训练数据已成功保存至 `{log_file}`。")
print(f"🏆 【自动存档结论】：历史最低验证集损失为 {best_val_loss:.4f}，最佳模型权重已自动锁存在第 **{best_epoch}** 轮。")
print(f"💡 报告撰写建议：如果第 {best_epoch} 轮之后 Val Loss 停止下降或开始反弹，则该点即为完美的过拟合拐点。")

# ==========================================
# 📊 阶段 4：载入历史最佳权重进行最终评估
# ==========================================
print(f"\n🕒 阶段 4：正在加载第 {best_epoch} 轮的最佳模型进行终极评估...")
model.load_state_dict(torch.load('best_lstm_model.pth'))
model.eval() 

all_preds = []
all_targets = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        predictions = model(batch_X)
        preds = (predictions > 0.5).int().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(batch_y.numpy())

print("\n📊 === LSTM 深度学习模型 (历史最佳版) 最终成绩单 ===")
print(f"Accuracy (准确率): {accuracy_score(all_targets, all_preds):.4f}")
print("\n分类报告:")
print(classification_report(all_targets, all_preds, digits=4))
print("\n混淆矩阵:")
print(confusion_matrix(all_targets, all_preds))