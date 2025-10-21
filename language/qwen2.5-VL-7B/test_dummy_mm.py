import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import asyncio
import os  # Import the OS module for exiting
from transformers import AutoProcessor
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TextPrompt

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# --- (The first part of your script is perfect, no changes needed) ---
def create_dummy_image(size=(200, 150), color="blue", text="Img 1"):
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=40)
    except IOError:
        font = ImageFont.load_default(size=30)
    draw.text((10, 10), text, fill="white", font=font)
    return img

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    return img_base64

image1 = create_dummy_image(color="blue", text="Blue")
image2 = create_dummy_image(color="red", text="Red")
base64_image1 = encode_image_to_base64(image1)
base64_image2 = encode_image_to_base64(image2)
images = [base64_image1, base64_image2]

question = "What are the colors of these two images?"
placeholders = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img}"}} for b64img in images]
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]},
]
processor = AutoProcessor.from_pretrained(MODEL_NAME)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
multi_image_prompt: TextPrompt = {
    "prompt": prompt,
    "multi_modal_data": {"image": [base64_image1, base64_image2]}
}

print("Initializing vLLM engine...")
engine_args = AsyncEngineArgs(model=MODEL_NAME, limit_mm_per_prompt={"image": 2})
engine = AsyncLLMEngine.from_engine_args(engine_args)
sampling_params = SamplingParams(max_tokens=100)
request_id = "multi-image-request-123"

async def main():
    print("Submitting generation request...")
    results_generator = engine.generate(
        prompt=multi_image_prompt,
        sampling_params=sampling_params,
        request_id=request_id
    )
    final_output = None
    async for request_output in results_generator:
        if request_output.finished:
            final_output = request_output
            print("Request finished.")

    if final_output:
        prompt = final_output.prompt
        generated_text = final_output.outputs[0].text
        print(f"\nPrompt: {prompt}\n")
        print(f"Generated Text: {generated_text}")

    # --- DEFINITIVE SHUTDOWN SEQUENCE ---
    print("Shutting down the engine...")
    engine.shutdown()

    # Give the shutdown signal a moment to propagate
    await asyncio.sleep(1)

    # Force exit the process with a success code to bypass the faulty finalizer
    print("Forcing exit to prevent finalization error...")
    os._exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        # This is expected. We can pass to allow the script to exit quietly.
        pass