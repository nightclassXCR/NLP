from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import logging
import os
from contextlib import asynccontextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存储模型
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    if not load_model():
        raise Exception("模型加载失败，服务无法启动")
    yield
    # 关闭时清理资源（如果需要）

# 初始化 FastAPI
app = FastAPI(
    title="Medical Chatbot API", 
    description="Qwen 医疗问答接口", 
    version="1.0",
    lifespan=lifespan
)

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

def load_model():
    """加载模型函数"""
    global tokenizer, model
    
    # 检查是否有微调后的模型
    finetuned_path = "./medical_qwen_finetuned"
    if os.path.exists(finetuned_path):
        logger.info("🚀 正在加载微调后的模型...")
        MODEL_NAME = finetuned_path
    else:
        logger.info("🚀 正在加载基础模型...")
        MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型，使用更兼容的参数
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        # 根据设备选择加载方式
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "cpu"
            
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
        model.eval()
        
        logger.info("✅ 模型加载完成！")
        logger.info(f"设备: {model.device}")
        logger.info(f"数据类型: {model.dtype}")
        return True
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        return False


@app.post("/ask")
async def ask_medical_bot(query: Query):
    """处理医学问答请求"""
    try:
        if not model or not tokenizer:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        if not query.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")
        
        # 构建对话消息
        messages = [
            {"role": "system", "content": "你是一名专业医生，请用简明、准确的方式回答医学问题。回答要专业但易懂，并提醒用户这只是参考建议，不能替代医生诊断。"},
            {"role": "user", "content": query.question}
        ]

        # 使用chat template处理对话
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码回答
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回答部分
        if "assistant" in full_response.lower():
            answer = full_response.split("assistant")[-1].strip()
        else:
            answer = full_response
        
        # 清理回答
        answer = answer.replace("AI回答：", "").strip()
        
        return {
            "success": True,
            "answer": answer,
            "question": query.question
        }
        
    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理问题失败: {str(e)}")

@app.get("/")
async def root():
    return {"message": "医学 Chatbot API 正在运行 🚀", "status": "running"}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None,
        "message": "服务运行正常"
    }

@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    return {
        "model_name": model.config.name_or_path if hasattr(model.config, 'name_or_path') else "Unknown",
        "device": str(model.device),
        "dtype": str(model.dtype),
        "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "Unknown"
    }

# 启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

