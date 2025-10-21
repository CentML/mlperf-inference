from datasets import get_dataset_config_names
from datasets import load_dataset
from PIL import Image
import base64
import io
import json
import ast

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

configs = get_dataset_config_names("MMMU/MMMU")

def format_list_with_letters(input_string):
  """
  Prepends each item in a list with a letter (A, B, C, ...) and a period,
  and then joins them into a single comma-separated string.

  Args:
    input_list: A list of strings.

  Returns:
    A single string with the formatted and joined items.
  """
  try:
    input_list = ast.literal_eval(input_string)

  except (ValueError, SyntaxError):
    print("Error: The input string is not a valid list format.")

  formatted_items = []
  for i, item in enumerate(input_list):
    # Generate the letter corresponding to the index (0 -> 'A', 1 -> 'B', etc.)
    letter = chr(65 + i)
    formatted_items.append(f"{letter}. {item}")

  # Join the formatted items with a comma and a space
  if not formatted_items:
     return ""
  return ", ".join(formatted_items)

# -----------------------------------------------------------------------------
# 1. Helper function to encode an image to base64
# -----------------------------------------------------------------------------
def pil_to_base64(image: Image.Image, format="jpeg") -> str:
    """Converts a PIL Image object to a base64 string with data URI."""
    # Handle images with alpha channels (transparency) when converting to JPEG
    if format.upper() == 'JPEG' and image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Format for OpenAI API is a data URI
    return f"data:image/{format.lower()};base64,{img_str}"

# -----------------------------------------------------------------------------
# 2. Main function to format a dataset example for the OpenAI API
# -----------------------------------------------------------------------------
def format_for_openai_api(example: dict, category: str) -> list:
    """
    Takes a dataset example and formats it into the OpenAI messages format
    for a multimodal request.
    """
    images = [] 
    # --- Step A: Find and encode all images ---
    for i in range(1, 9):
        image_key = f"image_{i}"
        if image_key in example and example[image_key] is not None:
            pil_image = example[image_key]
            base64_image = pil_to_base64(pil_image)
            
            images.append(base64_image)
    # --- Step B: Clean the question text ---
    original_question = example['question']
    options = format_list_with_letters(example['options'])
    if options:
      answer_options = f""" 1. Be concise and your answer should be only one of these possible options:\n{options}, 
                and your answer should start with the keywords: Final Response, and the letter you choose needs to be enclosed in square brackets []."""
    else:
      answer_options = """1. Be concise and answer and your answer should start with the keyword: Final Response and the answer should be enclosed in square brackets [].
                      If the answer is a number, put the number in the square brackets [] without any extra symbol or letter,
                      If the answer is a letter, just put the letter in the square brackets [],
                      If the answer is a word, just put the word in the square brackets [].
                      """
       
    prompt = f"""{original_question}.To answer the question follow these instructions:\n
                {answer_options}\n
                2. Avoid being verbose and avoid explaining your reasoning, just answer the question."""
    answer = example['answer']
    
    return {
        "prompt": prompt,
        "id": example["id"],
        "images": images,
        "answer": answer,
        "category": category
    }

def main():
    database = []
    for config_name in configs:
        dataset = load_dataset("MMMU/MMMU",config_name ,split='validation')
        print(f"len of {config_name} is: {len(dataset)}")
        database.extend([format_for_openai_api(dataset[i], config_name) for i in range(len(dataset))])

    with open("mmmu_data.json","w") as f:
        json.dump(database, f, indent=4)

if __name__ == "__main__":
    main()