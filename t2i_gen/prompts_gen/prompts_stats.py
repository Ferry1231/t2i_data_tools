import json
import matplotlib.pyplot as plt

# 修改为你的JSON文件路径
json_path = "/root/fengyuan/datasets/vision_auto_rubric/prompts/train_t.json"

def main():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lengths = [len(item.get("prompt", "").split()) for item in data if len(item.get("prompt", "").split()) <= 500]

    # 打印每条记录的长度
    for i, l in enumerate(lengths, 1):
        print(f"{i:03d}: {l} words")
    # 统计汇总
    if lengths:
        print("\n--- Summary ---")
        print(f"Total samples: {len(lengths)}")
        print(f"Average length: {sum(lengths) / len(lengths):.2f}")
        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")

    # --- 可视化 ---
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Prompt Lengths")
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("prompt_lengths_distribution_test_re.png")

if __name__ == "__main__":
    main()
