import sys, os
for _e in os.listdir(f"{venv_path}/lib"):
    if _e.startswith("python"):
        sys.path.insert(0, f"{venv_path}/lib/{_e}/site-packages")
        break

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

model = Qwen3VLModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()
ids = torch.tensor([input_ids], dtype=torch.long)
mask = (ids != pad_token_id).long()
with torch.no_grad():
    out = model.language_model(input_ids=ids)
hidden = out.last_hidden_state
seq_len = mask.shape[1]
last_pos = mask.flip([1]).argmax(dim=1)
col = -last_pos + (seq_len - 1)
emb = hidden[0, col[0], :]
norm = emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
result = (emb / norm).float().tolist()
