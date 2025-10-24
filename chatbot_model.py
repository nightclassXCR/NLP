# chatbot_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

class MedicalChatbot:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"🚀 正在加载模型 {model_name} ……（第一次运行会自动下载）")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",           # 自动选择CPU/GPU
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        print("✅ 模型加载完成！")

    def ask(self, question: str) -> str:
        """
        输入医学问题，返回AI回答。
        """
        system_prompt = (
            "你是一位医学AI助手，擅长解释症状、疾病、药物和健康管理问题。"
            "回答要清晰、专业、易懂，但不能替代医生诊断。"
        )

        full_prompt = f"{system_prompt}\n\n用户问题：{question}\n\nAI回答："

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 截取模型回答部分
        if "AI回答：" in answer:
            answer = answer.split("AI回答：")[-1]

        answer += "\n\n⚠️ 声明：本内容仅供医学学习与参考，不能替代医生诊断。"
        return answer


if __name__ == "__main__":
    bot = MedicalChatbot()
    while True:
        q = input("\n👩‍⚕️ 请输入医学问题（输入exit退出）：\n> ")
        if q.strip().lower() == "exit":
            break
        print("\n🩺 AI回答：", bot.ask(q))
