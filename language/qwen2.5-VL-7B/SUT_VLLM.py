import asyncio
import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoProcessor
import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


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

        self.model_path = model_path or f"Qwen/Qwen2.5-VL-7B-Instruct"

        if not batch_size:
            batch_size = 1
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."

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
        self.sampling_params = SamplingParams(**gen_kwargs)
        # self.sampling_params.all_stop_token_ids.add(self.model.get_tokenizer().eos_token_id)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(self.model_path)

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


# class SUTServer(SUT):
#     def __init__(
#         self,
#         model_path=None,
#         dtype="bfloat16",
#         total_sample_count=13368,
#         dataset_path=None,
#         batch_size=None,
#         workers=1,
#         tensor_parallel_size=8
#     ):

#         super().__init__(
#             model_path=model_path,
#             dtype=dtype,
#             total_sample_count=total_sample_count,
#             dataset_path=dataset_path,
#             workers=workers,
#             tensor_parallel_size=tensor_parallel_size,
#         )
#         self.request_id = 0

#         self.first_token_queue = queue.Queue()

#     def start(self):
#         # Create worker threads
#         for j in range(self.num_workers):
#             worker = threading.Thread(target=self.process_queries)
#             worker.start()
#             self.worker_threads[j] = worker

#     async def stream_output(self, qitem, results_generator):
#         first = True
#         async for request_output in results_generator:
#             output_response = request_output
#             if first:
#                 first_tokens = list(output_response.outputs[0].token_ids)
#                 response_data = array.array(
#                     "B", np.array(first_tokens, np.int32).tobytes())
#                 bi = response_data.buffer_info()
#                 response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
#                 lg.FirstTokenComplete(response)
#                 first = False

#         outputs = output_response
#         pred_output_tokens = list(output_response.outputs[0].token_ids)
#         n_tokens = len(pred_output_tokens)
#         response_array = array.array(
#             "B", np.array(pred_output_tokens, np.int32).tobytes()
#         )
#         bi = response_array.buffer_info()
#         response = [
#             lg.QuerySampleResponse(
#                 qitem.id,
#                 bi[0],
#                 bi[1],
#                 n_tokens)]
#         lg.QuerySamplesComplete(response)

#     def process_queries(self):
#         """Processor of the queued queries. User may choose to add batching logic"""
#         while True:

#             qitem = self.query_queue.get()
#             if qitem is None:
#                 break

#             input_ids_tensor = TokensPrompt(
#                 prompt_token_ids=self.data_object.input_ids[qitem.index])

#             # TODO: This PoC is super slow with significant overhead. Best to
#             # create a patch to `generate`
#             results_generator = self.model.generate(
#                 prompt=input_ids_tensor, sampling_params=self.sampling_params, request_id=str(
#                     self.request_id)
#             )
#             self.request_id += 1
#             asyncio.run(self.stream_output(qitem, results_generator))

#     def issue_queries(self, query_samples):
#         self.query_queue.put(query_samples[0])

#     def stop(self):
#         for _ in range(self.num_workers):
#             self.query_queue.put(None)

#         for worker in self.worker_threads:
#             worker.join()

#         self.first_token_queue.put(None)
#         self.ft_response_thread.join()

#     def load_model(self):
#         log.info("Loading model")
#         self.engine_args = AsyncEngineArgs(
#             self.model_path,
#             dtype=self.dtype,
#             tensor_parallel_size=self.tensor_parallel_size)
#         self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
#         log.info("Loaded model")

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

        # This will be the single, long-running asyncio event loop
        self.event_loop = None
        self.event_loop_thread = None
        
        # We'll use an asyncio.Queue to communicate between the issue_queries thread
        # and our main async event loop.
        self.async_query_queue = None


    def start(self):
        """Starts the asyncio event loop in a dedicated thread."""
        log.info("Starting SUT Server...")
        self.event_loop_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.event_loop_thread.start()


    def run_async_loop(self):
        """The target for our dedicated thread. It sets up and runs the event loop."""
        log.info("Asyncio event loop thread started.")
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)

        # Create the asyncio queue inside the loop it belongs to
        self.async_query_queue = asyncio.Queue()

        # Run the main async setup and worker creation
        self.event_loop.run_until_complete(self.async_startup())
        self.event_loop.run_forever()
        log.info("Asyncio event loop thread finished.")

    async def async_startup(self):
        """Initializes async resources like the engine and creates worker tasks."""
        
        # FIX: Call the synchronous load_model method.
        # It's a blocking call, which is acceptable here since it's part of the
        # one-time setup process before the server starts taking requests.
        self.load_model()
        
        # The rest of the async setup remains the same
        self.async_workers = [
            asyncio.create_task(self.process_queries_async()) for _ in range(self.num_workers)
        ]
        log.info(f"Started {self.num_workers} async worker tasks.")


    async def process_queries_async(self):
        """
        This is the new async worker. It replaces the old threaded `process_queries`.
        It runs in a continuous loop on the main event loop.
        """
        while True:
            qitem = await self.async_query_queue.get()
            try:
                if qitem is None:
                # Received poison pill, exit the loop.
                # The `finally` block will still execute.
                    break
                
                request_id = str(self.request_id_counter)
                self.request_id_counter += 1

                question = self.data_object.prompts[qitem.index]
                
                placeholders = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}} for b64img in self.data_object.images[qitem.index]]
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]},
                ]
                
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                multi_image_prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": self.data_object.images[qitem.index]}
}

                results_generator = self.model.generate(
                    prompt=multi_image_prompt, 
                    sampling_params=self.sampling_params, 
                    request_id=request_id
                )
                
                # --- The logic from your old `stream_output` is now here ---
                first = True
                output_response = None
                async for request_output in results_generator:
                    output_response = request_output
                    if first and output_response.outputs[0].token_ids:
                        first_tokens = list(output_response.outputs[0].token_ids)
                        response_data = array.array("B", np.array(first_tokens, np.int32).tobytes())
                        bi = response_data.buffer_info()
                        response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                        lg.FirstTokenComplete(response)
                        first = False

                if output_response:
                    # After the loop, process the final result
                    final_tokens = list(output_response.outputs[0].token_ids)
                    n_tokens = len(final_tokens)
                    response_array = array.array("B", np.array(final_tokens, np.int32).tobytes())
                    bi = response_array.buffer_info()
                    response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)]
                    lg.QuerySamplesComplete(response)

            except Exception as e:
                log.error(f"Error processing item {qitem.id if qitem else 'None'}: {e}")
            finally:
                # CRITICAL FIX: This ensures task_done() is called for every item,
                # including the `None` poison pill that signals the end.
                self.async_query_queue.task_done()



    def issue_queries(self, query_samples):
        """
        This is called by the synchronous MLPerf thread.
        We must use a thread-safe method to submit work to our event loop.
        """
        # We need to handle the case where the loop isn't ready yet
        if not self.event_loop:
            time.sleep(0.1) # Simple wait for the loop to start
            if not self.event_loop:
                 log.error("Event loop not available to issue queries.")
                 return

        # Use run_coroutine_threadsafe to safely put an item into the asyncio.Queue
        # from this synchronous thread.
        # This is the bridge between the two concurrency models.
        for sample in query_samples:
            asyncio.run_coroutine_threadsafe(
                self.async_query_queue.put(sample), self.event_loop
            )

    def stop(self):
        """Stops the workers, the vLLM engine, and the event loop in the correct order."""
        log.info("Stopping SUT server...")

        if self.event_loop and self.async_query_queue:
            # Step 1 & 2: Signal our workers and wait for them to finish.
            for _ in range(self.num_workers):
                asyncio.run_coroutine_threadsafe(self.async_query_queue.put(None), self.event_loop)
            
            join_future = asyncio.run_coroutine_threadsafe(self.async_query_queue.join(), self.event_loop)
            join_future.result()
            
            # Step 3: Explicitly shut down the vLLM engine.
            if self.model:
                log.info("Shutting down vLLM engine...")
                self.model.shutdown()
                log.info("vLLM engine shut down.")

            # --- FINAL FIX IS HERE ---
            # Step 4: Add a final cleanup phase to gracefully cancel any remaining
            # background tasks (including potentially orphaned vLLM tasks).
            
            # This coroutine will gather and cancel all running tasks.
            async def final_cleanup(loop):
                tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task(loop=loop)]
                if tasks:
                    log.info(f"Cleaning up {len(tasks)} remaining tasks...")
                    for task in tasks:
                        task.cancel()
                    # We await the gather to allow the tasks to process their cancellation.
                    await asyncio.gather(*tasks, return_exceptions=True)
                log.info("Final cleanup complete.")

            # Schedule the cleanup on the loop and wait for it to finish.
            cleanup_future = asyncio.run_coroutine_threadsafe(final_cleanup(self.event_loop), self.event_loop)
            cleanup_future.result()

            # Step 5: Now that everything is clean, stop the event loop.
            if self.event_loop.is_running():
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                
        # Step 6: Wait for the event loop's thread to terminate.
        if self.event_loop_thread:
            self.event_loop_thread.join()
            
        log.info("SUT server stopped.")

    def load_model(self):
        """
        A synchronous method to load the vLLM AsyncEngine.
        This replaces the empty `load_model` from the parent class.
        """
        log.info("Loading model...")
        self.engine_args = AsyncEngineArgs(
            model=self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt={"image":9}
        )
        
        # FIX: `from_engine_args` is a synchronous call, so `await` is removed.
        # This will block until the engine is fully initialized.
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Model loaded.")

    # You would need to make a small change to the base SUT __init__
    # to accept `_load_model=False` to prevent it from calling the old sync `load_model`.
    # def load_model(self):
    #     """The old sync load_model is no longer used by SUTServer."""
    #     pass