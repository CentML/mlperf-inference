import json
import numpy as np
from pathlib import Path
from dataset import Dataset

filepath = Path("/home/jcalderon/MLCommons/inference/language/qwen2.5-VL-7B/output/mlperf_log_accuracy.json")

target_path = Path("/home/jcalderon/MLCommons/inference/language/qwen2.5-VL-7B/datasets/mmmu_data.json")

with open(filepath, "r") as f:
    data = json.load(f)

sorted_list = sorted(data, key=lambda item: item["qsl_idx"])
target = Dataset(dataset_path=target_path)
ids = target.ids
targets = target.targets

for i in range(len(sorted_list)):
    text_encoding = np.frombuffer(bytes.fromhex(sorted_list[i]["data"]), np.int32)
    pred_txt = "".join(chr(j) for j in text_encoding)
    qsl_idx = sorted_list[i]["qsl_idx"]
    id = ids[qsl_idx]
    answer = targets[qsl_idx]
    print("===========")
    print(f"qsl_idx :{qsl_idx}")
    print(f"id      : {id}")
    print(f"answer  : {answer}")
    print(f"pred    :{pred_txt}")
    print("==========\n")