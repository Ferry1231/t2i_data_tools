import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# 修改为你的JSON文件路径
json_path = "/root/fengyuan/datasets/HPDv3/test_rewritten.json"

with open(json_path, "r") as f:
    file = json.load(f)

print(file[12]['prompt'], '\n', file[12]['prompt_fault'])

# for i, item in enumerate(tqdm(file)):
#     item['path1'] = None
#     item['path2'] = None
#     item['model1'] = None
#     item['model2'] = None
#     item['key'] = f"rubric_{i}"

# with open(json_path, "w", encoding="utf-8") as f:
#     json.dump(file, f, ensure_ascii=False, indent=4)