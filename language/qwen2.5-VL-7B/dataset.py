import random
import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io

# import utils
import copy
import pickle

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MMMU-Dataset")


class Dataset:
    def __init__(
        self,
        total_sample_count=13368,
        perf_count_override=None,
        dataset_path=None,
    ):
        self.dataset_path = dataset_path

        # self.total_sample_count = total_sample_count
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.prompts), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warning(
                "Processed pickle file {} not found. Please check that the path is correct".format(
                    self.dataset_path
                )
            )

        log.info("Loading dataset...")
        import pandas as pd

        self.processed_data = pd.read_json(self.dataset_path)

        self.prompts = self.processed_data.prompt.tolist()
        self.ids = self.processed_data.id.tolist()
        self.images = self.processed_data.images.tolist()
        self.categories = self.processed_data.category.tolist()
        self.targets = self.processed_data.answer.tolist()
        del self.processed_data
        log.info("Finished loading dataset.")

    def postProcess(
        self,
        out_tokens,
        query_id_list=None,
        sample_index_list=None,
    ):
        """Postprocesses output prediction"""

        # TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        # Everything is padded to max_len (1024), so prune the input and parse
        # to numpy
        output_seq = out_tokens
        assert len(query_id_list) == len(output_seq)

        return [np.asarray(out, dtype=np.int32) for out in output_seq]

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass
