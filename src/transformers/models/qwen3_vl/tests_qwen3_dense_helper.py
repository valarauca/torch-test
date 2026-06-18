import sys, os
for _e in os.listdir(f"{venv_path}/lib"):
    if _e.startswith("python"):
        sys.path.insert(0, f"{venv_path}/lib/{_e}/site-packages")
        break

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path)
model = Qwen3VLModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.eval()
ids = torch.tensor([input_ids], dtype=torch.long)
with torch.no_grad():
    out = model.language_model(input_ids=ids)
result = out.last_hidden_state[0, -1, :].float().tolist()
