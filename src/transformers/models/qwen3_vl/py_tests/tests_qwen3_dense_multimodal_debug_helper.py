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
device = "cuda"
model = Qwen3VLModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device)
model.eval()

gt, gh, gw = grid_thw
pixel_values = torch.tensor(pixel_values_flat, dtype=torch.float32).reshape(gt * gh * gw, -1).bfloat16().to(device)
image_grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.long, device=device)
ids = torch.tensor([[image_token_id] * num_tokens], dtype=torch.long, device=device)
mm_token_type_ids = torch.ones_like(ids)

from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
import torch.nn.functional as _F
with torch.no_grad():
    bi, bw = get_vision_bilinear_indices_and_weights(image_grid_thw, num_grid_per_side=model.visual.num_grid_per_side, spatial_merge_size=model.visual.spatial_merge_size)
    _pe_t = (model.visual.pos_embed(bi) * bw[:, :, None]).sum(0)
    pos_embeds_raw = _pe_t.float().flatten().tolist()
    _patch_t = model.visual.patch_embed(pixel_values)
    patch_embed_out = _patch_t.float().flatten().tolist()
    _hidden = _patch_t + _pe_t.to(_patch_t.dtype)
    hidden_after_pe = _hidden.float().flatten().tolist()
    _pos_ids = get_vision_position_ids(image_grid_thw, spatial_merge_size=model.visual.spatial_merge_size)
    _rotary = model.visual.rotary_pos_emb(_pos_ids)
    _emb = torch.cat((_rotary, _rotary), dim=-1)
    _cu = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    _cu = _F.pad(_cu, (1, 0), value=0)
    after_block0 = model.visual.blocks[0](_hidden, cu_seqlens=_cu, position_embeddings=(_emb.cos(), _emb.sin())).float().flatten().tolist()
    # Cross-check: run Python block0 on Rust's EXACT hidden_pe/cos/sin (bit-identical inputs)
    _rh = torch.tensor(rust_hidden_pe, dtype=torch.float32).reshape(total_patches, -1).bfloat16().to(device)
    _rc = torch.tensor(rust_cos, dtype=torch.float32).reshape(total_patches, -1).to(device)
    _rs = torch.tensor(rust_sin, dtype=torch.float32).reshape(total_patches, -1).to(device)
    after_block0_shared = model.visual.blocks[0](_rh, cu_seqlens=_cu, position_embeddings=(_rc, _rs)).float().flatten().tolist()

# Step 1: vision encoder output
with torch.no_grad():
    vis_out = model.visual(pixel_values, grid_thw=image_grid_thw)
    pre_merger = vis_out.last_hidden_state.float().flatten().tolist()
    image_features = vis_out.pooler_output.float().flatten().tolist()

# Step 2: mRoPE position IDs
position_ids, _ = model.get_rope_index(ids, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids)
position_ids = position_ids.flatten().tolist()

# Step 3: full model hidden states
with torch.no_grad():
    out = model(input_ids=ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw, mm_token_type_ids=mm_token_type_ids)
hidden = out.last_hidden_state
hidden_states = hidden.float().flatten().tolist()

# Step 4: pool last non-padding token
mask = (ids != pad_token_id).long()
seq_len = mask.shape[1]
last_pos = mask.flip([1]).argmax(dim=1)
col = -last_pos + (seq_len - 1)
emb = hidden[0, col[0], :]
pooled = emb.float().flatten().tolist()

# Step 5: L2 normalize
norm = emb.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
embedding = (emb / norm).float().tolist()
