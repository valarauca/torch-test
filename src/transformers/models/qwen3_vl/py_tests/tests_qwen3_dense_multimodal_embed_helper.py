import sys, os
for _e in os.listdir(f"{venv_path}/lib"):
    if _e.startswith("python"):
        sys.path.insert(0, f"{venv_path}/lib/{_e}/site-packages")
        break

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

tokenizer = AutoTokenizer.from_pretrained(model_path)
pad_token_id = tokenizer.eos_token_id
model = Qwen3VLModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()

gt, gh, gw = grid_thw
pixel_values = torch.tensor(pixel_values_flat, dtype=torch.float32).reshape(gt * gh * gw, -1).bfloat16()
image_grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.long)
ids = torch.tensor([input_ids], dtype=torch.long)
mm_token_type_ids = torch.ones_like(ids)

with torch.no_grad():
    out = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids)
hidden = out.last_hidden_state
mask = (ids != pad_token_id).long()
seq_len = mask.shape[1]
last_pos = mask.flip([1]).argmax(dim=1)
col = -last_pos + (seq_len - 1)
emb = hidden[0, col[0], :]
norm = emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
result = (emb / norm).float().tolist()
