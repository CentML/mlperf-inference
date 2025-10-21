import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
import evaluate
import nltk
from dataset import Dataset

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoTokenizer.from_pretrained(MODEL)

results = Path("./output/mlperf_log_accuracy.json")
target_path = Path("./datasets/mmmu_data.json")

with open(results, "r") as f:
    data = json.load(f)

target = Dataset(
    dataset_path=target_path
)

ids = target.ids
targets = target.targets

pred_token_ids = [np.frombuffer(bytes.fromhex(data[i]["data"]), np.int32) for i in range(len(data))]

pred_txt = processor.batch_decode(pred_token_ids, skip_special_tokens=True)

for i in range(len(data)):
    qsl_idx = data[i]["qsl_idx"]
    id = ids[qsl_idx]
    answer = targets[qsl_idx]
    print("===========")
    print(f"qsl_idx :{qsl_idx}")
    print(f"id      : {id}")
    print(f"answer  : {answer}")
    print(f"pred    :{pred_txt[i]}")
    print("==========\n")
##############################
##############################

# from dataset import Dataset
# import os
# import time
# import numpy as np
# import json
# import nltk
# import array
# import torch
# from torch.nn.functional import pad
# from torch.utils.data import DataLoader
# import evaluate
# import argparse
# import nltk
# from transformers import AutoModelForCausalLM, AutoTokenizer


# def get_args():
#     """Parse commandline."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json"
#     )
#     parser.add_argument(
#         "--dataset-file",
#         required=True,
#         help="path to cnn_eval.json")
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="verbose messages")
#     parser.add_argument(
#         "--dtype",
#         default="int64",
#         help="dtype of the accuracy log",
#         choices=["int32", "int64"],
#     )
#     parser.add_argument(
#         "--model-name",
#         default="Qwen/Qwen2.5-VL-7B-Instruct",
#         help="Model name")
#     parser.add_argument(
#         "--total-sample-count",
#         default=13368,
#         type=int,
#         help="Model name")
#     args = parser.parse_args()
#     return args


# def postprocess_text(preds, targets):
#     preds = [pred.strip() for pred in preds]
#     targets = [target.strip() for target in targets]

#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

#     return preds, targets


# def main():

#     args = get_args()
#     model_name = args.model_name
#     dataset_path = args.dataset_file
#     total_sample_count = args.total_sample_count
#     metric = evaluate.load("rouge")
#     nltk.download("punkt")
#     nltk.download('punkt_tab')

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name,
#         padding_side="left",
#     )
#     tokenizer.pad_token = tokenizer.eos_token
#     data_object = Dataset(
#         dataset_path=dataset_path,
#         total_sample_count=total_sample_count,
#     )

#     targets = data_object.targets

#     with open(args.mlperf_accuracy_file, "r") as f:
#         results = json.load(f)

#     # Deduplicate the results loaded from the json
#     dedup_results = []
#     seen = set()
#     for result in results:
#         item = result["qsl_idx"]
#         if item not in seen:
#             seen.add(item)
#             dedup_results.append(result)
#     results = dedup_results

#     target_required = []
#     preds_token_ids = []

#     eval_dtype = np.int64
#     if args.dtype == "int32":
#         eval_dtype = np.int32

#     for pred in results:
#         qsl_idx = pred["qsl_idx"]
#         target = targets[qsl_idx]
#         target_required.append(target)
#         preds_token_ids.append(
#             np.frombuffer(
#                 bytes.fromhex(
#                     pred["data"]),
#                 eval_dtype))

#     preds_decoded_text = tokenizer.batch_decode(
#         preds_token_ids, skip_special_tokens=True
#     )

#     preds, targets = postprocess_text(preds_decoded_text, target_required)

#     result = metric.compute(
#         predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
#     )
#     result = {k: f"{round(np.mean(v) * 100, 4)}" for k, v in result.items()}
#     prediction_lens = [len(pred) for pred in preds]
#     result["gen_len"] = np.sum(prediction_lens)
#     result["gen_num"] = len(preds)
#     print("\nResults\n")
#     print(result)


# if __name__ == "__main__":
#     main()
