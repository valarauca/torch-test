import sys, os
for _e in os.listdir(f"{venv_path}/lib"):
    if _e.startswith("python"):
        sys.path.insert(0, f"{venv_path}/lib/{_e}/site-packages")
        break

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

device = "cuda"
lm = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
model = lm.model
model.eval()
processor = AutoProcessor.from_pretrained(model_path)

token_yes = processor.tokenizer.get_vocab()["yes"]
token_no = processor.tokenizer.get_vocab()["no"]
score_linear = torch.nn.Linear(lm.lm_head.weight.shape[1], 1, bias=False).to(device).to(model.dtype)
with torch.no_grad():
    score_linear.weight[0] = lm.lm_head.weight.data[token_yes] - lm.lm_head.weight.data[token_no]

img = Image.open(image_path).convert("RGB")
messages = [
    {"role": "system", "content": [{"type": "text", "text": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'}]},
    {"role": "user", "content": [
        {"type": "text", "text": "<Instruct>: Given a search query, retrieve relevant candidates that answer the query."},
        {"type": "text", "text": "<Query>:"},
        {"type": "text", "text": query_text},
        {"type": "text", "text": "\n<Document>:"},
        {"type": "image", "image": img},
    ]},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=[img], return_tensors="pt").to(device)

with torch.no_grad():
    hidden = model(**inputs).last_hidden_state[:, -1]
    py_score = torch.sigmoid(score_linear(hidden)).squeeze(-1).item()

input_ids = inputs["input_ids"].flatten().cpu().tolist()
pixel_values_flat = inputs["pixel_values"].float().flatten().cpu().tolist()
grid_thw = inputs["image_grid_thw"][0].cpu().tolist()
