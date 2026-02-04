import json
import random
import os
from pathlib import Path


def split_json_file(
    input_json_path: str,
    output_dir: str,
    num_splits: int = 3,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    将一个 JSON 文件随机划分成若干份并保存到指定目录。

    Args:
        input_json_path (str): 原始 JSON 文件路径
        output_dir (str): 输出文件夹路径
        num_splits (int): 要划分的份数
        shuffle (bool): 是否随机打乱数据
        seed (int): 随机种子，保证可复现
    """

    # 1. 读取原始 JSON 文件
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(len(data))

    # 2. 可选：随机打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    # 3. 计算每份大小
    total = len(data)
    chunk_size = total // num_splits
    remainder = total % num_splits

    # 4. 创建输出文件夹
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 5. 分割并保存
    start = 0
    for i in range(num_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = data[start:end]
        start = end

        output_path = os.path.join(output_dir, f"split_{i+1}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(chunk)} samples to {output_path}")

    print(f"\n🎉 Done! {total} items split into {num_splits} parts.")


if __name__ == "__main__":
    # 示例调用
    split_json_file(
        input_json_path="/root/fengyuan/datasets/HPDv3/test_rewritten.json",
        output_dir="/root/fengyuan/datasets/vision_auto_rubric/for_rubrics/positive",
        num_splits=1,  # 指定分成几份
    )
