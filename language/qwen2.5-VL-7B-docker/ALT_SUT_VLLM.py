import asyncio
import os
import time
import numpy as np
import array
# import torch
# from torch.nn.functional import pad
# from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
# from vllm.inputs import TokensPrompt
# from transformers import AutoProcessor
# import pickle
import time
import threading
# import tqdm
import queue

import logging
# from typing import TYPE_CHECKING, Optional, List
# from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset
import sys
import subprocess
import requests
from contextlib import suppress
import signal
from openai import OpenAI, AsyncOpenAI
from typing import Any, Dict


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Qwen2.5-VL-7B")

# ---------- Config ----------
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
HOST = os.environ.get("VLLM_HOST", "vllm")
PORT = int(os.environ.get("VLLM_PORT", "8000"))

# Extra vLLM server args if you need them (GPU/CPU flags, trust-remote-code, tensor-parallel-size, etc.)
EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "--trust-remote-code").split()

BASE_URL = f"http://{HOST}:{PORT}/v1"
HEALTH_URLS = [
    f"http://{HOST}:{PORT}/health",         # preferred if available
    f"http://{HOST}:{PORT}/v1/models",      # fallback readiness check
]

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
        _load_model=False
    ):
        self.proc = None
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

        if _load_model: self.load_model()
        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 1024,
        }
        self.max_tokens = 1024
        self.temperature = 0.0
        # self.sampling_params = SamplingParams(**gen_kwargs)
        self.sampling_params = gen_kwargs
        # self.sampling_params.all_stop_token_ids.add(self.model.get_tokenizer().eos_token_id)

        self.num_workers = workers
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
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            # input_text_tensor = [
            #     self.data_object.input[q.index] for q in qitem]
            # for in_text in input_text_tensor:
            #     log.info(f"Input: {in_text}")

            tik2 = time.time()
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor, sampling_params=self.sampling_params
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
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def load_model(self):
        log.info("Loading model...")
        # self.model = LLM(
        #     self.model_path,
        #     dtype=self.dtype,
        #     tensor_parallel_size=self.tensor_parallel_size,
        # )
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

    def start_vllm_server(self, model: str, host: str, port: int, extra_args: list[str]) -> subprocess.Popen:
        """
        Launch vLLM's OpenAI-compatible server as a subprocess.
        Returns a Popen handle.
        """
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", host,
            "--port", str(port),
        ] + extra_args

        # Inherit stdout/stderr so you can see logs in your terminal.
        # (If you prefer, redirect to a file or PIPE.)
        print("Launching vLLM server:\n ", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        return proc

    def wait_until_ready(self, timeout_s: int = 300, poll_interval_s: float = 0.5) -> None:
        """
        Poll one or more health endpoints until HTTP 200, or raise on timeout.
        """
        start = time.time()
        last_err = None
        while time.time() - start < timeout_s:
            for url in HEALTH_URLS:
                try:
                    r = requests.get(url, timeout=2)
                    if r.status_code == 200:
                        # For /v1/models, ensure JSON is present (indicates API is fully up)
                        if url.endswith("/v1/models"):
                            with suppress(Exception):
                                _ = r.json()
                        print(f"Server ready at {url}")
                        return
                except Exception as e:
                    last_err = e
            time.sleep(poll_interval_s)

        raise TimeoutError(f"vLLM server didn't become ready within {timeout_s}s. Last error: {last_err}")


    def terminate_process(self, proc: subprocess.Popen, grace_s: int = 15) -> None:
        """
        Try to stop the server gracefully, then force-kill if needed.
        Cross-platform friendly.
        """
        if proc.poll() is not None:
            return  # already exited

        try:
            # POSIX: try SIGINT first (clean shutdown), then SIGTERM, then SIGKILL.
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=grace_s)
                return
            except subprocess.TimeoutExpired:
                pass

            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        finally:
            with suppress(Exception):
                proc.wait(timeout=2)


    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        # ... same arguments as before
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=8
    ):
        # We call a modified super().__init__ that doesn't load the model yet
        # because model loading needs to be async.
        # This is a bit of a simplification; you might need to adjust the base SUT init.
        # For this example, let's assume the base init can be called without loading the model.
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
            # Add a flag to skip model loading in the base class constructor
            _load_model=False 
        )
        self.request_id_counter = 0
        client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key="EMPTY"
        )
        self._client = client
        # This will be the single, long-running asyncio event loop
        self.event_loop = None
        self.event_loop_thread = None
        
        # We'll use an asyncio.Queue to communicate between the issue_queries thread
        # and our main async event loop.
        self.async_query_queue = None


    def start(self):
        # self.proc = self.start_vllm_server(MODEL, HOST, PORT, EXTRA_ARGS)
        # self.wait_until_ready()

        # # Optional: print the models list to confirm we're talking to the right thing
        # r = requests.get(f"{BASE_URL}/models", timeout=3)
        # print("\n[Models]", r.json())
        pass


    async def _issue_one(
        self,
        sample: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        log.info("CALLED _issue_one")
        """Send one streaming chat.completion request and record timings."""

        contents = [{"type": "text", "text": self.data_object.prompts[sample.index]}]
        for img_b64 in self.data_object.images[sample.index]:
            contents.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })

        messages = [{"role": "user", "content": contents}]

        params = dict(
            model=self.model_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        async with semaphore:
            ttft_set = False

            # await the async creation; ask for a streaming iterator
            stream = await self._client.chat.completions.create(
                stream=True,
                messages=messages,
                **params
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
                        text_int32 = np.array([ord(c) for c in text], dtype=np.int32)
                        response_data = array.array("B", text_int32.tobytes())
                        bi = response_data.buffer_info()
                        response = [lg.QuerySampleResponse(sample.id, bi[0], bi[1])]
                        lg.FirstTokenComplete(response)
                        ttft_set = True
                    out.append(text)

            # when the stream ends, total latency   
            final_tokens = "".join(out)
            final_tokens_int32 = np.array([ord(c) for c in final_tokens], dtype=np.int32)
            n_tokens = len(final_tokens_int32)
            response_array = array.array("B", final_tokens_int32.tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(sample.id, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)


    async def _issue_queries_async(self, query_samples):
        """Async internal version used by the sync wrapper."""
        log.info(f"CALLED _issue_queries_async, num workers: {self.num_workers}")
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
        # if self.proc is not None:
        #     print("\nShutting down vLLM server…")
        #     self.terminate_process(self.proc)
        #     print("Done.")
        pass