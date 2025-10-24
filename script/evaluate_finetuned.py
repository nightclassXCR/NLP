from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluate import load

# 模型路径（使用微调后的）
MODEL_PATH = "./finetuned_qwen_medical"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.float16)

# 加载评测集
data = load_dataset("json", data_files="data/eval_data.json")["train"]

# 指标
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

preds, refs = [], []

# 推理生成
for i, sample in enumerate(data):
    inputs = f"指令：{sample['instruction']}\n回答："
    inputs_tokenized = tokenizer(inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs_tokenized,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去掉prompt前缀
    answer = answer.split("回答：")[-1].strip()

    preds.append(answer)
    refs.append(sample["output"])

# 自动评测
bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
rouge_score = rouge.compute(predictions=preds, references=refs)
bert_score = bertscore.compute(predictions=preds, references=refs, lang="zh")

print("\n📊 模型评测结果：")
print(f"BLEU: {bleu_score['bleu']:.4f}")
print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
print(f"BERTScore-F1: {sum(bert_score['f1']) / len(bert_score['f1']):.4f}")
print("✅ 评测完成！")
