import os
import json
import time
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("请先安装openai库: pip install openai")
    exit(1)


class DebateAgent:
    def __init__(self, name: str, api_key: str, base_url: str, model: str):
        self.name = name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.history: List[Dict[str, str]] = []

    def set_system_prompt(self, role: str):
        self.history = [{"role": "system", "content": role}]

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def generate_response(self, topic: str, round_num: int, max_tokens: int = 2000) -> str:
        prompt = f"""
你正在参与一场学术辩论，议题是：{topic}

当前是第 {round_num} 轮。请基于之前的对话历史，提出你的观点和论据。

要求：
1. 观点明确，逻辑清晰
2. 引用相关理论或证据
3. 针对对方观点提出反驳或补充
4. 语言专业但易懂
5. 长度控制在合理范围内

请直接输出你的辩论内容，不要添加开场白或结束语。
"""
        self.add_message("user", prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            self.add_message("assistant", content)
            return content
        except Exception as e:
            print(f"[{self.name}] API调用失败: {e}")
            return f"[API调用失败] {str(e)}"


def run_debate(topic: str, n_rounds: int = 5):
    print("=" * 80)
    print(f"学术辩论开始")
    print(f"议题: {topic}")
    print(f"轮次: {n_rounds}")
    print("=" * 80 + "\n")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    glm_key = os.getenv("GLM_API_KEY")

    if not deepseek_key:
        print("错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        return

    glm_role = """
你是GLM-5.2，一位逻辑严谨的AI研究员。
你的职责是提出创新性的观点，进行严密的逻辑推理，审查对方论点的合理性。
"""

    deepseek_role = """
你是DeepSeek-V4-Pro，一位富有洞察力的AI创新架构师。
你的职责是提出创新的算法框架和理论推导，关注创新性和整体架构设计。
"""

    glm_agent = DebateAgent("GLM-5.2", glm_key or "dummy", "https://open.bigmodel.cn/api/paas/v4/", "glm-5.2")
    deepseek_agent = DebateAgent("DeepSeek-V4-Pro", deepseek_key, "https://api.deepseek.com/v1", "deepseek-chat")

    glm_agent.set_system_prompt(glm_role)
    deepseek_agent.set_system_prompt(deepseek_role)

    debate_history = []

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*80}")
        print(f"第 {round_num}/{n_rounds} 轮")
        print(f"{'='*80}")

        print(f"\n[GLM-5.2] 发言:")
        glm_response = glm_agent.generate_response(topic, round_num)
        print(glm_response[:500] + "..." if len(glm_response) > 500 else glm_response)
        debate_history.append({"round": round_num, "speaker": "GLM-5.2", "content": glm_response})

        deepseek_agent.add_message("user", f"对方(GLM-5.2)的观点:\n{glm_response}")

        print(f"\n[DeepSeek-V4-Pro] 发言:")
        deepseek_response = deepseek_agent.generate_response(topic, round_num)
        print(deepseek_response[:500] + "..." if len(deepseek_response) > 500 else deepseek_response)
        debate_history.append({"round": round_num, "speaker": "DeepSeek-V4-Pro", "content": deepseek_response})

        glm_agent.add_message("user", f"对方(DeepSeek-V4-Pro)的观点:\n{deepseek_response}")

        time.sleep(2)

    print("\n" + "=" * 80)
    print("辩论总结")
    print("=" * 80)

    summary_prompt = f"""
请对以下辩论进行总结：

议题: {topic}

辩论记录:
{json.dumps(debate_history, ensure_ascii=False, indent=2)}

请总结双方的核心观点、达成的共识、存在的分歧，以及最终建议。
"""

    glm_agent.set_system_prompt("你是一位资深学术评审专家，请对辩论进行客观总结。")
    glm_agent.add_message("user", summary_prompt)
    
    try:
        summary_response = glm_agent.client.chat.completions.create(
            model=glm_agent.model,
            messages=glm_agent.history,
            max_tokens=1500,
            temperature=0.5,
        )
        summary = summary_response.choices[0].message.content.strip()
        print("\n[GLM-5.2 总结]:")
        print(summary)
    except Exception as e:
        print(f"总结失败: {e}")

    with open(f"debate_result_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(debate_history, f, ensure_ascii=False, indent=2)
    print(f"\n辩论记录已保存到 debate_result_{int(time.time())}.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="跨模型学术辩论工具")
    parser.add_argument("--topic", type=str, required=True, help="辩论议题")
    parser.add_argument("--rounds", type=int, default=5, help="辩论轮次")
    args = parser.parse_args()

    run_debate(args.topic, args.rounds)
