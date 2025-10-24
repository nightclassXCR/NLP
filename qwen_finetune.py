from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("🚀 加载模型中……")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 补充 pad_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda:0",
    trust_remote_code=True
)
print("模型设备：", model.device)
print("✅ 模型加载完成")

# 加载并校验数据集
dataset = load_dataset("json", data_files="./data/filtered_data.json")
required_columns = ["instruction", "output"]
for col in required_columns:
    if col not in dataset["train"].column_names:
        raise ValueError(f"数据集缺少必要字段：{col}")
# 过滤空文本样本
dataset = dataset.filter(lambda x: len(x["instruction"].strip()) > 0 and len(x["output"].strip()) > 0)
print(f"过滤后数据集大小：{len(dataset['train'])} 条")

# 预处理函数（适配 batched=True）
def preprocess_function(examples):
    # 使用 Qwen 官方对话模板直接生成 token id 列表（已完成编码）
    chats_token_ids = []
    for ins, out in zip(examples["instruction"], examples["output"]):
        # 生成单样本的 token id 列表
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": ins}, {"role": "assistant", "content": out}],
            add_generation_prompt=False,
            truncation=True,  # 截断过长文本
            max_length=512,   # 限制最大长度
        )
        # 手动 padding 到 max_length（若不足则用 pad_token_id 填充）
        if len(token_ids) < 512:
            token_ids += [tokenizer.pad_token_id] * (512 - len(token_ids))
        # 确保不超过 max_length（防止截断不生效）
        token_ids = token_ids[:512]
        chats_token_ids.append(token_ids)

    # 直接构建 model_inputs（无需二次编码）
    model_inputs = {
        "input_ids": chats_token_ids,
        "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in seq] for seq in chats_token_ids]
    }

    # 批量处理 labels
    input_ids_batch = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
    batch_size, seq_len = input_ids_batch.shape
    labels_batch = torch.full((batch_size, seq_len), -100, dtype=torch.long)

    # 查找 <|assistant|> 的 token id 并掩码用户输入部分
    assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    if assistant_token_id == -1:
        raise ValueError("Tokenizer 未识别 <|assistant|> token，请更新 transformers 库和模型")

    assistant_mask = torch.tensor([[id == assistant_token_id for id in seq] for seq in input_ids_batch], dtype=torch.bool)

    for i in range(batch_size):
        pos = assistant_mask[i].nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            start_idx = pos[0] + 1
            if start_idx < seq_len:
                labels_batch[i, start_idx:] = input_ids_batch[i, start_idx:]

    model_inputs["labels"] = labels_batch.tolist()
    return model_inputs

print("🔧 预处理中……")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
print("✅ 预处理完成，共", len(tokenized_dataset["train"]), "条数据")

# 数据 Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=0.1,
    logging_steps=5,
    logging_first_step=True,
    save_strategy="no",
    optim="adamw_torch",
    fp16=False,
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=1.0,
       learning_rate=5e-6,  # 调低学习率
)

# 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# 保存模型
model.save_pretrained("./medical_qwen_finetuned")
tokenizer.save_pretrained("./medical_qwen_finetuned")
print("✅ 微调完成！模型已保存到 ./medical_qwen_finetuned")