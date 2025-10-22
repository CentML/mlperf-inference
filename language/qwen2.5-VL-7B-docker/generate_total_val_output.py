import json
import numpy as np
from pathlib import Path
from dataset import Dataset
import re
import os
import argparse
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoTokenizer.from_pretrained(MODEL)
# The regular expression pattern to find content inside square brackets
# each answer will have the structure:
# text1 = "Final Response [D]"
pattern = r"\[(.*?)\]"

filepath = Path(os.getcwd(), "./output/mlperf_log_accuracy.json")
target_path = Path(os.getcwd(), "./datasets/mmmu_data.json")

def main(mode: str):
    with open(filepath, "r") as f:
        data = json.load(f)

    sorted_list = sorted(data, key=lambda item: item["qsl_idx"])
    target = Dataset(dataset_path=target_path)
    ids = target.ids
    store_data = dict()

    decoded_output = [np.frombuffer(bytes.fromhex(sorted_list[i]["data"]), np.int32) for i in range(len(sorted_list))]
    if mode == "offline":
        text_outputs = processor.batch_decode(decoded_output, skip_special_tokens=True) 
    else:
        text_outputs = ["".join(chr(j) for j in text_integers) for text_integers in decoded_output]
    for i in range(len(sorted_list)):
        pred_txt = text_outputs[i]
        qsl_idx = sorted_list[i]["qsl_idx"]
        id = ids[qsl_idx]
        text_extract = re.search(pattern, pred_txt)
        cleaned_answer = text_extract.group(1) if text_extract else ""
        store_data[id] = cleaned_answer

    with open("total_val_output.json", "w") as f:
        json.dump(store_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="offline")
    args = parser.parse_args()
    main(args.mode)
