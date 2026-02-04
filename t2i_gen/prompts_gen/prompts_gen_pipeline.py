import json
import random
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# 配置
# -------------------------------
input_json_path = "/root/fengyuan/datasets/HPDv3/train.json"   
output_json_path = "/root/fengyuan/datasets/HPDv3/train_rewritten.json"  

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
# 定义单条改写函数
# -------------------------------
def rewrite_prompt(item):
    original_prompt = item.get("prompt", "")
    target_words = random.randint(10, 200)
    chat_prompt = prompts_template.format(TARGET_WORDS=target_words, PROMPT=original_prompt)
    
    try:
        response = client.chat.completions.create(
            model="/root/fengyuan/models/Qwen3-8B",
            messages=[{"role": "user", "content": chat_prompt}],
            max_tokens=30000,
            temperature=3.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        rewritten_prompt = response.choices[0].message.content.strip()
    except Exception as e:
        print("Error rewriting prompt:", e)
        rewritten_prompt = original_prompt

    new_item = item.copy()
    new_item["prompt"] = rewritten_prompt
    return new_item

# -------------------------------
# 多线程执行
# -------------------------------
new_data = []
max_workers = 16  # 可以根据你的机器和 API 并发能力调整

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(rewrite_prompt, item): idx for idx, item in enumerate(data)}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Rewriting Prompts"):
        new_data.append(future.result())

# -------------------------------
# 保存新的 JSON 文件
# -------------------------------
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print("✅ 所有 prompts 改写完成，已保存至:", output_json_path)
