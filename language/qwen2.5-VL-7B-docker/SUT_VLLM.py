import asyncio
import os
import time
import numpy as np
import array
import time
import threading
import queue

import logging
import contextlib
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
        total_sample_count=13368,
        dataset_path=None,
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8,
    ):
        self.model_path = model_path or f"Qwen/Qwen2.5-VL-7B-Instruct"

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

        self._client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key="EMPTY"
        )

    def start(self):
        pass

    def stop(self):
        pass

    async def _issue_one(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send one streaming chat.completion request and record timings."""

        contents = [
            {"type": "text", "text": self.data_object.prompts[sample.index]}]
        for img_b64 in self.data_object.images[sample.index]:
            contents.append({
                "type": "image_url",
                "image_url": {"url": img_b64}
            })

        messages = [{"role": "user", "content": contents}]

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
        # Decide which context manager to use
        tasks = [self._issue_one(s) for s in query_samples]
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
        log.info(f"Sending {len(query_samples)} queries samples")
        asyncio.run(self._issue_queries_async(query_samples))

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
        workers=1,
        tensor_parallel_size=8,
    ):
        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
        )

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
            sample = await self.async_query_queue.get()
            try:
                if sample is None:
                # Received poison pill, exit the loop.
                # The `finally` block will still execute.
                    break
                
                contents = [
                    {"type": "text", "text": self.data_object.prompts[sample.index]}]
                for img_b64 in self.data_object.images[sample.index]:
                    contents.append({
                        "type": "image_url",
                        "image_url": {"url": img_b64}
                    })

                messages = [{"role": "user", "content": contents}]

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

            except Exception as e:
                log.error(f"Error processing item {sample.id if sample else 'None'}: {e}")
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
            
            # Step 3: Add a final cleanup phase to gracefully cancel any remaining
            
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

            # Step 4: Now that everything is clean, stop the event loop.
            if self.event_loop.is_running():
                self.event_loop.call_soon_threadsafe(self.event_loop.stop)
                
        # Step 5: Wait for the event loop's thread to terminate.
        if self.event_loop_thread:
            self.event_loop_thread.join()
            
        log.info("SUT server stopped.")