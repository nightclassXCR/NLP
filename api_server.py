from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

# 初始化 FastAPI
app = FastAPI(title="Medical Chatbot API", description="Qwen 医疗问答接口", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或者改成 ["http://localhost:3000"] 等前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求体
class Query(BaseModel):
    question: str

# 加载模型（只加载一次）
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
print("🚀 正在加载模型，请稍等……")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
print("✅ 模型加载完成！")

@app.post("/ask")
def ask_medical_bot(query: Query):
    messages = [
        {"role": "system", "content": "你是一名专业医生，请用简明、准确的方式回答医学问题。"},
        {"role": "user", "content": query.question}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.2
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}

@app.get("/")
async def root():
    return {"message": "医学 Chatbot API 正在运行 🚀"}

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

