import json
import random
from openai import OpenAI
from tqdm import tqdm

# -------------------------------
# 配置
# -------------------------------
input_json_path = "/root/fengyuan/datasets/HPDv3/test.json"   # 原始 JSON 文件路径
output_json_path = "/root/fengyuan/datasets/HPDv3/test_rewritten.json"  # 保存新文件路径

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# -------------------------------
# 改写模板（保持你的 prompt 不变）
# -------------------------------
prompts_template = """
你是一名提示词改写助手。
你的任务是：在不改变原始语义和风格的前提下，将给定的 prompt 改写至目标词数 {TARGET_WORDS}。

---

【要求】
* 不要改动 prompt 的主要主题或语气。
* 可以通过加入/减少更多细节、形容词、环境描述、氛围词或艺术元素来达到目标词数。
* 保持语言自然、流畅。
* 不要输出任何解释或多余说明，只输出改写后的文本。
* 输出文本的词数尽量接近 {TARGET_WORDS}。

---

【输入】
{PROMPT}

---

请生成一个词数为 {TARGET_WORDS} 的版本。
"""

# -------------------------------
# 读取 JSON 数据
# -------------------------------
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------
# 遍历每条记录并改写 prompt
# -------------------------------
new_data = []
for idx, item in enumerate(tqdm(data, desc="Rewriting Prompts")):
    original_prompt = item.get("prompt", "")
    
    # 随机生成目标词数
    target_words = random.randint(5, 100)
    
    # 构造改写 prompt
    chat_prompt = prompts_template.format(TARGET_WORDS=target_words, PROMPT=original_prompt)
    
    # 调用 vLLM/OpenAI API
    try:
        response = client.chat.completions.create(
            model="/root/fengyuan/models/Qwen3-8B",
            messages=[{"role": "user", "content": chat_prompt}],
            max_tokens=30000,
            temperature=3.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        rewritten_prompt = response.choices[0].message.content.strip()
        # print(f"[{idx}/{len(data)}] Original: {len(original_prompt.split())} words → Rewritten: {len(rewritten_prompt.split())} words")
    except Exception as e:
        print(f"[{idx}/{len(data)}] Error rewriting prompt:", e)
        rewritten_prompt = original_prompt  # 出错则保留原始 prompt

    # 保存改写后的记录
    new_item = item.copy()
    new_item["prompt"] = rewritten_prompt
    new_data.append(new_item)

# -------------------------------
# 保存新的 JSON 文件
# -------------------------------
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("✅ 所有 prompts 改写完成，已保存至:", output_json_path)
