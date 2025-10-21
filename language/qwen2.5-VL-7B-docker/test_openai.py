# import openai
# import json
# from pathlib import Path

# # --- Configuration ---
# # 1. Set the base URL of your vLLM server
# VLLM_BASE_URL = "http://localhost:8080/v1"

# # 2. Set the model name. This should match the model loaded on your vLLM server.
# #    (e.g., "llava-hf/llava-1.5-7b-hf")
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct" 

# # 3. For local vLLM, the API key is often not required, but the library expects a value.
# API_KEY = "not-needed" 

# path_to_file = "./datasets/mmmu_data.json"
# fileloc = Path(path_to_file)

# def main():
#     with open(fileloc, "r") as f:
#         data = json.load(f)

#     item = data[0]

#     content_parts = []
#     content_parts.append({
#         "type": "text",
#         "text": item['prompt']
#     })

#     for img64 in item["images"]:
#         content_parts.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": img64
#             }
#         })

#     try:
#         print("Initializing OpenAI client for vLLM...")
#         # Point the client to your local vLLM server
#         client = openai.OpenAI(
#             base_url=VLLM_BASE_URL,
#             api_key=API_KEY,
#         )
#         # Send the request to the chat completions endpoint
#         response = client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": content_parts,
#                 }
#             ],
#             max_tokens=256, # Adjust as needed
#             temperature=0.0, # Adjust as needed
#             top_p=1,
#             seed=42,
#         )

#         # Print the response from the model
#         print("\n--- Model Response ---")
#         print(response.choices[0].message.content)
#         print("---------------------\n")

#     except openai.APIConnectionError as e:
#         print("Connection Error: Could not connect to the vLLM server.")
#         print(f"Please ensure the server is running at {VLLM_BASE_URL} and is accessible.")
#         print(f"Details: {e.__cause__}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     main()

## ================================================== #####
## ================================================== #####

# import openai
# import json
# import time
# from pathlib import Path

# VLLM_BASE_URL = "http://localhost:8080/v1"
# MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# API_KEY = "not-needed"
# path_to_file = "./datasets/mmmu_data.json"
# fileloc = Path(path_to_file)

# def main():
#     with open(fileloc, "r") as f:
#         data = json.load(f)
#     item = data[0]

#     content_parts = [{"type": "text", "text": item["prompt"]}]
#     for img64 in item.get("images", []):
#         content_parts.append({"type": "image_url", "image_url": {"url": img64}})

#     print("Initializing OpenAI client for vLLM...")
#     client = openai.OpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

#     params = dict(
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": content_parts}],
#         max_tokens=256,
#         temperature=0.0,
#         top_p=1,
#         seed=42,
#         stream=True,  # <- classic generator streaming
#     )

#     start = time.perf_counter()
#     ttft = None
#     first_token = None
#     out_parts = []

#     try:
#         # Classic streaming generator (works with vLLM’s OpenAI-compatible endpoint)
#         stream = client.chat.completions.create(**params)
#         for chunk in stream:
#             # Defensive checks: some chunks only carry role/metadata
#             choices = getattr(chunk, "choices", None)
#             if not choices:
#                 continue

#             delta = choices[0].delta
#             text = getattr(delta, "content", None)

#             if text:
#                 if ttft is None:
#                     ttft = time.perf_counter() - start
#                     first_token = text
#                     print(f"[TTFT {ttft*1000:.1f} ms] first token: {repr(first_token)}")
#                 print(text, end="", flush=True)
#                 out_parts.append(text)

#             # Optional: handle finish signal
#             if getattr(choices[0], "finish_reason", None) is not None:
#                 # finish_reason may appear on the last chunk
#                 pass

#         total = time.perf_counter() - start
#         final_text = "".join(out_parts)

#         print("\n\n--- Stats ---")
#         print("TTFT:", f"{ttft:.3f} s" if ttft is not None else "(no token received)")
#         print("First token:", repr(first_token))
#         print(f"End-to-end: {total:.3f} s")
#         print(f"Output chars: {len(final_text)}")
#         print("-----------------------")
#         print("\nFinal output:\n", final_text)

#         # If you also want token counts, do a non-streamed call after (vLLM returns usage):
#         # usage_resp = client.chat.completions.create(
#         #     **{k:v for k,v in params.items() if k != "stream"}
#         # )
#         # print("usage:", usage_resp.usage)  # prompt_tokens, completion_tokens, total_tokens

#     except openai.APIConnectionError as e:
#         print("Connection Error: Could not connect to the vLLM server.")
#         print(f"Ensure the server is running at {VLLM_BASE_URL}. Details: {e.__cause__}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# if __name__ == "__main__":
#     main()


## ==================================================== ##
## ==================================================== ##


import json
import time
import asyncio
from pathlib import Path
from openai import AsyncOpenAI

VLLM_BASE_URL = "http://vllm:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
API_KEY = "not-needed"

path_to_file = "./datasets/mmmu_data.json"
fileloc = Path(path_to_file)

def normalize_image_url(u: str) -> str:
    # If it's already an http(s) or data URL, keep it.
    if u.startswith("http://") or u.startswith("https://") or u.startswith("data:"):
        return u
    # If it looks like raw base64, prefix a default mime
    # (adjust to image/jpeg if your dataset uses JPEGs).
    return "data:image/png;base64," + u

async def run_one(client: AsyncOpenAI, messages, **kwargs):
    start = time.perf_counter()
    ttft = None
    first_token = None
    out = []

    try:
        stream = await client.chat.completions.create(stream=True, messages=messages, **kwargs)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            text = getattr(delta, "content", None)

            if text:
                if ttft is None:
                    ttft = time.perf_counter() - start
                    first_token = text
                    print(f"[TTFT {ttft*1000:.1f} ms] first token: {repr(first_token)}")
                print(text, end="", flush=True)
                out.append(text)

        total = time.perf_counter() - start
        return {
            "ttft": ttft,
            "first_token": first_token,
            "e2e": total,
            "text": "".join(out),
        }

    except Exception as e:
        # vLLM (and the SDK) often attach a useful JSON body on e
        # but to keep it simple, just print e
        print("\n[run_one] Exception:", repr(e))
        raise

async def main_async():
    # Load one sample
    with open(fileloc, "r") as f:
        data = json.load(f)
    item = data[0]

    # Build multimodal content
    content_parts = [{"type": "text", "text": item["prompt"]}]
    for img in item.get("images", []):
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": normalize_image_url(img)}
        })

    # IMPORTANT: messages must be a list (no trailing comma)
    messages = [{"role": "user", "content": content_parts}]

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=API_KEY)

    params = dict(
        model=MODEL_NAME,
        max_tokens=256,
        temperature=0.0,
        top_p=1,
        seed=42,
        # Do NOT pass "stream" here; it's given in run_one
    )

    result = await run_one(client, messages, **params)

    print("\n\n--- Stats ---")
    print("TTFT:", f"{result['ttft']:.3f} s" if result["ttft"] is not None else "(no token received)")
    print("First token:", repr(result["first_token"]))
    print(f"End-to-end: {result['e2e']:.3f} s")
    print(f"Output chars: {len(result['text'])}")
    print("-----------------------")
    print("\nFinal output:\n", result["text"])

if __name__ == "__main__":
    asyncio.run(main_async())
