import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

# ==========================================
# ⚙️ 阶段 0：核心配置与日志记录器
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 6
LR_LIST = [1e-5, 2e-5, 3e-5]
BATCH_SIZE = 16

# 动态获取当前路径，设置日志与结果文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "clean_data.csv")
log_file = os.path.join(current_dir, "F_bert_experiment_log.txt")
result_file = os.path.join(current_dir, "F_bert_result.txt")

# 初始化（清空）日志和结果文件
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("=== BERT 多学习率消融实验训练日志 ===\n")
with open(result_file, 'w', encoding='utf-8') as f:
    f.write("=== BERT 多学习率消融实验 最终成绩单 ===\n")

# 定义双写打印函数（同时输出到控制台和 txt 日志）
def print_and_log(text, file_path=log_file):
    print(text)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(text + "\n")

print_and_log(f"🚀 BERT 消融实验引擎启动！设备: {device}")

# ==========================================
# 🛠️ 阶段 1：数据准备 (只做一次，节省时间)
# ==========================================
print_and_log("\n🕒 阶段 1：读取原始数据与 BERT 分词处理...")
df = pd.read_csv(data_path)
text_column = 'cleaned' 
df = df.dropna(subset=[text_column, 'label'])

X_raw = df[text_column].values
y_raw = df['label'].values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print_and_log("⚙️ 正在将文本编码为 BERT 矩阵...")
encoded_data = tokenizer(
    X_raw.tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=64,
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(y_raw, dtype=torch.long)

inputs_temp, inputs_test, masks_temp, masks_test, y_temp, y_test = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2, random_state=42
)
inputs_train, inputs_val, masks_train, masks_val, y_train, y_val = train_test_split(
    inputs_temp, masks_temp, y_temp, test_size=0.125, random_state=42
)

train_loader = DataLoader(TensorDataset(inputs_train, masks_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(inputs_val, masks_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(inputs_test, masks_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

print_and_log(f"📊 数据划分完成：训练集 {len(inputs_train)} | 验证集 {len(inputs_val)} | 测试集 {len(inputs_test)}")

# ==========================================
# 🔥 阶段 2：网格搜索大循环 (核心炼丹区)
# ==========================================
for lr in LR_LIST:
    print_and_log(f"\n" + "="*50)
    print_and_log(f"⚔️ 正在启动实验组：学习率 LR = {lr}")
    print_and_log(f"="*50)
    
    # 【极其关键】：每次必须重新初始化模型，彻底洗脑，防止跨实验污染！
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_val_accuracy = 0
    best_epoch = 0
    model_save_path = os.path.join(current_dir, f'best_bert_lr_{lr}.pth')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            if step % 200 == 0 and not step == 0:
                print(f"  > [LR: {lr}] Batch {step:>5,} of {len(train_loader):>5,}...")
                
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        model.eval()
        total_val_accuracy = 0
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                preds = torch.argmax(outputs.logits, dim=1).flatten()
                total_val_accuracy += (preds == b_labels.flatten()).cpu().numpy().mean()
                
        avg_val_accuracy = total_val_accuracy / len(val_loader)
        epoch_time = time.time() - start_time
        
        epoch_log = f"🌟 Epoch {epoch+1}/{EPOCHS} | 耗时: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_accuracy:.4f}"
        print_and_log(epoch_log)
        
        # 自动保存当前学习率下的最佳模型
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_save_path)
            print_and_log(f"   [!] 发现新高！模型已锁定并保存至 {model_save_path}")

    print_and_log(f"\n🏆 [LR = {lr}] 训练结束！最佳模型锁定在第 {best_epoch} 轮 (验证集 Acc: {best_val_accuracy:.4f})")

    # ==========================================
    # 📊 阶段 3：当前学习率的测试集评估
    # ==========================================
    print_and_log(f"🕒 正在加载 {model_save_path} 进行测试集评估...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(b_labels.cpu().numpy())
            
    final_acc = accuracy_score(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # 按照 LSTM 的严格格式写入 result.txt
    result_text = f"\n=== BERT (LR={lr}) 测试集最终成绩单 ===\n"
    result_text += f"Accuracy: {final_acc:.4f}\n\n"
    result_text += f"Classification Report:\n{class_report}\n"
    result_text += f"Confusion Matrix:\n{conf_matrix}\n"
    
    # 打印到控制台，并追加写入结果文件
    print(result_text)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(result_text + "\n")

print_and_log("\n🎉 所有消融实验全部运行完毕！请查收您的 log 和 result 文件。")