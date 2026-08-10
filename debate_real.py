import os
import json
import time
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("请先安装openai库: pip install openai")
    exit(1)


class DeepSeekDebater:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def debate(self, topic: str, debate_history: List[Dict], round_num: int) -> str:
        history_str = "\n".join([f"{d['speaker']}: {d['content'][:200]}..." for d in debate_history[-4:]]) if debate_history else "无"
        
        prompt = f"""
你是DeepSeek-V4-Pro，一位富有洞察力的AI创新架构师。

学术辩论议题：{topic}

当前是第 {round_num} 轮。

之前的辩论记录：
{history_str}

请基于以上历史，提出你的观点和论据。要求：
1. 观点明确，逻辑清晰
2. 引用相关理论或证据
3. 针对对方观点提出反驳或补充
4. 语言专业但易懂
5. 控制在500-800字

直接输出你的辩论内容。
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[DeepSeek API调用失败] {str(e)}"


def run_debate_with_glm(topic: str, n_rounds: int = 5):
    print("=" * 80)
    print(f"GLM-5.2 vs DeepSeek-V4-Pro 真实学术辩论")
    print(f"议题: {topic}")
    print(f"轮次: {n_rounds}")
    print("=" * 80 + "\n")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("错误: 请先设置环境变量 DEEPSEEK_API_KEY")
        print("PowerShell: $env:DEEPSEEK_API_KEY='your_key_here'")
        print("然后再次运行此脚本")
        return

    deepseek = DeepSeekDebater(deepseek_key)
    debate_history: List[Dict] = []

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*80}")
        print(f"第 {round_num}/{n_rounds} 轮")
        print(f"{'='*80}")

        print("\n[GLM-5.2] 请输入你的观点：")
        glm_input = input("> ").strip()
        if glm_input.lower() in ["quit", "exit", "stop"]:
            print("辩论终止")
            break

        debate_history.append({
            "round": round_num,
            "speaker": "GLM-5.2",
            "content": glm_input
        })

        print("\n[DeepSeek-V4-Pro] 正在思考...")
        deepseek_response = deepseek.debate(topic, debate_history, round_num)
        print(f"\n[DeepSeek-V4-Pro] 的回应：")
        print(deepseek_response)

        debate_history.append({
            "round": round_num,
            "speaker": "DeepSeek-V4-Pro",
            "content": deepseek_response
        })

        time.sleep(1)

    print("\n" + "=" * 80)
    print("辩论结束")
    print("=" * 80)

    timestamp = int(time.time())
    output_file = f"debate_result_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(debate_history, f, ensure_ascii=False, indent=2)
    print(f"\n辩论记录已保存到: {output_file}")

    return debate_history


def run_debate_auto(topic: str, n_rounds: int = 5):
    print("=" * 80)
    print(f"GLM-5.2 vs DeepSeek-V4-Pro 自动学术辩论")
    print(f"议题: {topic}")
    print(f"轮次: {n_rounds}")
    print("=" * 80 + "\n")

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        print("错误: 请先设置环境变量 DEEPSEEK_API_KEY")
        return

    deepseek = DeepSeekDebater(deepseek_key)
    debate_history: List[Dict] = []

    glm_role = "你是GLM-5.2，一位逻辑严谨的AI研究员。请针对议题提出观点和论据。"

    try:
        from openai import OpenAI as GLMOpenAI
        glm_key = os.getenv("GLM_API_KEY")
        if glm_key:
            glm_client = GLMOpenAI(api_key=glm_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
            use_glm_api = True
        else:
            print("未设置GLM_API_KEY，使用模拟模式（仅展示DeepSeek观点）")
            use_glm_api = False
    except:
        use_glm_api = False

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*80}")
        print(f"第 {round_num}/{n_rounds} 轮")
        print(f"{'='*80}")

        if use_glm_api:
            print("\n[GLM-5.2] 正在思考...")
            history_str = "\n".join([f"{d['speaker']}: {d['content'][:150]}..." for d in debate_history[-3:]])
            
            glm_prompt = f"""
你是GLM-5.2，一位逻辑严谨的AI研究员。

学术辩论议题：{topic}

第 {round_num} 轮。

之前的辩论：
{history_str}

请提出你的观点和论据（500字以内）：
"""
            glm_response = glm_client.chat.completions.create(
                model="glm-5.2",
                messages=[{"role": "user", "content": glm_prompt}],
                max_tokens=1000,
                temperature=0.7,
            ).choices[0].message.content.strip()
        else:
            glm_response = f"[GLM-5.2 第{round_num}轮观点占位] 需要设置GLM_API_KEY环境变量才能调用真实GLM API"

        print(f"\n[GLM-5.2] 发言：")
        print(glm_response)
        debate_history.append({"round": round_num, "speaker": "GLM-5.2", "content": glm_response})

        print("\n[DeepSeek-V4-Pro] 正在思考...")
        deepseek_response = deepseek.debate(topic, debate_history, round_num)
        print(f"\n[DeepSeek-V4-Pro] 回应：")
        print(deepseek_response)
        debate_history.append({"round": round_num, "speaker": "DeepSeek-V4-Pro", "content": deepseek_response})

        time.sleep(2)

    timestamp = int(time.time())
    output_file = f"debate_auto_result_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(debate_history, f, ensure_ascii=False, indent=2)
    print(f"\n辩论记录已保存到: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLM-5.2 vs DeepSeek-V4-Pro 真实学术辩论工具")
    parser.add_argument("--topic", type=str, required=True, help="辩论议题")
    parser.add_argument("--rounds", type=int, default=5, help="辩论轮次")
    parser.add_argument("--auto", action="store_true", help="自动模式（需要GLM_API_KEY）")
    args = parser.parse_args()

    if args.auto:
        run_debate_auto(args.topic, args.rounds)
    else:
        run_debate_with_glm(args.topic, args.rounds)
