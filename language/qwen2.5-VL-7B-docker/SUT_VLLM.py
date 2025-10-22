import asyncio
import os
import time
import numpy as np
import array
import time
import threading
import queue

import logging

import mlperf_loadgen as lg
from dataset import Dataset
from openai import AsyncOpenAI
from typing import Any, Dict


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Qwen2.5-VL-7B")

# ---------- Config ----------
HOST = os.environ.get("VLLM_HOST", "vllm")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://{HOST}:{PORT}/v1"


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        use_cached_outputs=False,
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8,
        scenario="offline"
    ):
        self.model_path = model_path or f"Qwen/Qwen2.5-VL-7B-Instruct"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.num_workers = workers
        self.params = {
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        if scenario == "offline":
            from vllm import SamplingParams
            from transformers import AutoProcessor

            self.load_model()
            self.sampling_params = SamplingParams(**self.params)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.request_id_counter = 0

            self.worker_threads = [None] * self.num_workers
            self.query_queue = queue.Queue()

            self.use_cached_outputs = use_cached_outputs
            self.sample_counter = 0
            self.sample_counter_lock = threading.Lock()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitems = self.query_queue.get()
            if qitems is None:
                break

            query_ids = [q.index for q in qitems]

            tik1 = time.time()

            prompts = []
            for item in qitems:
                question = self.data_object.prompts[item.index]

                placeholders = [{"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64img}"}} for b64img in self.data_object.images[item.index]]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        *placeholders, {"type": "text", "text": question}]},
                ]

                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": self.data_object.images[item.index]}
                })

            tik2 = time.time()
            outputs = self.model.generate(
                prompts=prompts, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
                # log.info(f"Output: {output.outputs[0].text}")
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitems)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitems[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitems)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        from vllm import LLM
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=8,
        scenario="offline"
    ):
        super().__init__(
            model_path=model_path,
            batch_size=batch_size,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
            scenario=scenario
        )
        self._client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key="EMPTY"
        )

    def start(self):
        pass

    async def _issue_one(
        self,
        sample: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        log.info("CALLED _issue_one")
        """Send one streaming chat.completion request and record timings."""

        contents = [
            {"type": "text", "text": self.data_object.prompts[sample.index]}]
        for img_b64 in self.data_object.images[sample.index]:
            contents.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })

        messages = [{"role": "user", "content": contents}]

        async with semaphore:
            ttft_set = False

            # await the async creation; ask for a streaming iterator
            stream = await self._client.chat.completions.create(
                stream=True,
                messages=messages,
                model=self.model_path,
                **self.params
            )
            out = []
            # iterate asynchronously
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                # first non-empty token → TTFT
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    if ttft_set is False:
                        text_int32 = np.array([ord(c)
                                              for c in text], dtype=np.int32)
                        response_data = array.array("B", text_int32.tobytes())
                        bi = response_data.buffer_info()
                        response = [
                            lg.QuerySampleResponse(
                                sample.id, bi[0], bi[1])]
                        lg.FirstTokenComplete(response)
                        ttft_set = True
                    out.append(text)

            # when the stream ends, total latency
            final_tokens = "".join(out)
            final_tokens_int32 = np.array(
                [ord(c) for c in final_tokens], dtype=np.int32)
            n_tokens = len(final_tokens_int32)
            response_array = array.array("B", final_tokens_int32.tobytes())
            bi = response_array.buffer_info()
            response = [
                lg.QuerySampleResponse(
                    sample.id,
                    bi[0],
                    bi[1],
                    n_tokens)]
            lg.QuerySamplesComplete(response)

    async def _issue_queries_async(self, query_samples):
        """Async internal version used by the sync wrapper."""
        log.info(
            f"CALLED _issue_queries_async, num workers: {self.num_workers}")
        semaphore = asyncio.Semaphore(self.num_workers)
        tasks = [self._issue_one(s, semaphore) for s in query_samples]
        return await asyncio.gather(*tasks)

    def issue_queries(self, query_samples):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return asyncio.run_coroutine_threadsafe(
                self._issue_queries_async(query_samples), loop
            ).result()
        log.info("CALLED BEFORE ASYNCIO RUN:")
        asyncio.run(self._issue_queries_async(query_samples))

    def stop(self):
        pass
