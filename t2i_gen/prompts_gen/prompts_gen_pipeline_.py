prompts = """
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


from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="/root/fengyuan/models/Qwen3-8B",
    messages=[
        {"role": "user", "content": prompts.format(TARGET_WORDS=100, PROMPT="Two men in white shirts and red suspenders kiss under a blossoming tree.")},
    ],
    max_tokens=30000,
    temperature=3.0,
    # top_p=1.0,
    # presence_penalty=1.5,
    extra_body={
        # "top_k": 20, 
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
print("Chat response:", chat_response.choices[0].message.content)


