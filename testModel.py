from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载微调后的模型和 tokenizer
MODEL_PATH = "./medical_qwen_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 输入样例
instruction = "患者有发热和咳嗽，应该怎么办？"

# 编码输入
inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)

# 将输入移动到正确的设备（如果使用GPU）
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 生成模型输出
with torch.no_grad():  # 禁用梯度计算，节省内存
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,  # 设定输出的最大长度
        num_beams=5,     # 使用束搜索提高生成质量
        no_repeat_ngram_size=2,  # 防止生成重复的短语
        temperature=0.7,  # 控制生成文本的多样性
    )

# 解码输出文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印生成的答案
print(f"输入: {instruction}")
print(f"生成的答案: {generated_text}")
