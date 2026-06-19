pub mod dense;
pub mod moe;
pub mod vision;

#[cfg(test)]
mod tests_qwen3_dense;

use std::{collections::HashSet, future::Future, path::PathBuf, pin::Pin, sync::Arc};

use anyhow::{Context, Result};
use dense::{Attention, DecoderLayer, Linear, Mlp, RmsNorm, RotaryEmbedding, TextConfig, TextModel};
use moe::{Experts, LayerMlp, MoeDecoderLayer, MoeTextConfig, MoeTextModel, SparseMoeBlock, TopKRouter, is_moe_layer};
use tch::{Kind, Tensor};
use vision::{Conv3d, LayerNorm, PatchMerger, VisionAttention, VisionBlock, VisionConfig, VisionMlp, VisionModel, VisionPatchEmbed, VisionRotaryEmbedding};

use crate::tensors::SafeTensor;
#[cfg(test)]
use crate::transformers::repo::init_repo;
use crate::transformers::{
    model_ids::ModelIds,
    traits::{EmbeddingModel, EmbeddingScheme, ImageTokenizer, LocalModelBuilder, ModelFactory, ModelLoader, ModelRepo, RankingModel, TextTokenizer, TokenizedData},
};

// ---------------------------------------------------------------------------
// Weight map — searches across one or more safetensors shards
// ---------------------------------------------------------------------------

struct WeightMap {
    shards: Vec<SafeTensor>,
    device: tch::Device,
}

impl WeightMap {
    /// Loads either a single `model.safetensors` or all shards listed in
    /// `model.safetensors.index.json`.
    async fn load(repo: &dyn ModelRepo, device: tch::Device) -> Result<Self> {
        let mut shards = Vec::new();

        if let Some(path) = repo.get_local_path("model.safetensors").await? {
            shards.push(SafeTensor::load(&path)?);
        } else {
            let idx_path = repo
                .get_local_path("model.safetensors.index.json")
                .await?
                .context("neither model.safetensors nor model.safetensors.index.json found")?;

            let idx_str = std::fs::read_to_string(&idx_path)?;
            let idx: serde_json::Value = serde_json::from_str(&idx_str)?;
            let weight_map = idx["weight_map"].as_object().context("invalid index: missing weight_map")?;

            let shard_names: HashSet<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();

            for shard_name in shard_names {
                let path = repo.get_local_path(shard_name).await?.with_context(|| format!("shard not found: {shard_name}"))?;
                shards.push(SafeTensor::load(&path)?);
            }
        }

        Ok(Self { shards, device })
    }

    fn from_paths(paths: Vec<PathBuf>, device: tch::Device) -> Result<Self> {
        let shards = paths.iter().map(|p| SafeTensor::load(p)).collect::<Result<_>>()?;
        Ok(Self { shards, device })
    }

    /// Finds `name` across all shards and returns an independent copy on the target device.
    fn get(&self, name: &str) -> Result<Tensor> {
        for shard in &self.shards {
            if let Some(kind) = shard.kind_of(name) {
                let wrapper = shard.get_tensor(name, kind, tch::Device::Cpu)?.unwrap();
                let mut out = Tensor::empty(wrapper.size().as_slice(), (kind, self.device));
                out.copy_(&*wrapper);
                return Ok(out);
            }
        }
        anyhow::bail!("weight not found: {name}")
    }
}

// ---------------------------------------------------------------------------
// Shard path resolver — async half of weight loading
// ---------------------------------------------------------------------------

async fn resolve_shard_paths(repo: &dyn ModelRepo) -> Result<Vec<PathBuf>> {
    if let Some(path) = repo.get_local_path("model.safetensors").await? {
        return Ok(vec![path]);
    }
    let idx_path = repo
        .get_local_path("model.safetensors.index.json")
        .await?
        .context("neither model.safetensors nor model.safetensors.index.json found")?;
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(idx_path)?)?;
    let shard_names: HashSet<&str> = idx["weight_map"]
        .as_object()
        .context("invalid index: missing weight_map")?
        .values()
        .filter_map(|v| v.as_str())
        .collect();
    let mut paths = Vec::new();
    for name in shard_names {
        paths.push(repo.get_local_path(name).await?.with_context(|| format!("shard not found: {name}"))?);
    }
    Ok(paths)
}

// ---------------------------------------------------------------------------
// Shared construction helpers
// ---------------------------------------------------------------------------

fn linear(ws: &WeightMap, prefix: &str, has_bias: bool) -> Result<Linear> {
    Ok(Linear {
        weight: ws.get(&format!("{prefix}.weight"))?,
        bias: if has_bias { Some(ws.get(&format!("{prefix}.bias"))?) } else { None },
    })
}

fn rms_norm(ws: &WeightMap, prefix: &str, eps: f64) -> Result<RmsNorm> {
    Ok(RmsNorm {
        weight: ws.get(&format!("{prefix}.weight"))?,
        eps,
    })
}

fn layer_norm(ws: &WeightMap, prefix: &str) -> Result<LayerNorm> {
    Ok(LayerNorm {
        weight: ws.get(&format!("{prefix}.weight"))?,
        bias: ws.get(&format!("{prefix}.bias"))?,
        eps: 1.0e-6,
    })
}

fn patch_merger(ws: &WeightMap, prefix: &str, merged_size: i64, use_postshuffle_norm: bool) -> Result<PatchMerger> {
    Ok(PatchMerger {
        norm: layer_norm(ws, &format!("{prefix}.norm"))?,
        fc1: linear(ws, &format!("{prefix}.linear_fc1"), true)?,
        fc2: linear(ws, &format!("{prefix}.linear_fc2"), true)?,
        merged_size,
        use_postshuffle_norm,
    })
}

fn build_vision_model(ws: &WeightMap, cfg: &VisionConfig) -> Result<VisionModel> {
    let vp = "model.visual";
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let merged_size = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
    let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt() as i64;

    let patch_embed = VisionPatchEmbed {
        proj: Conv3d {
            weight: ws.get(&format!("{vp}.patch_embed.proj.weight"))?,
            bias: ws.get(&format!("{vp}.patch_embed.proj.bias"))?,
            kernel: [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
        },
        in_channels: cfg.in_channels,
        temporal_patch_size: cfg.temporal_patch_size,
        patch_size: cfg.patch_size,
        embed_dim: cfg.hidden_size,
    };

    let pos_embed = ws.get(&format!("{vp}.pos_embed.weight"))?;
    let rotary_emb = VisionRotaryEmbedding::new(head_dim / 2, ws.device);

    let mut blocks = Vec::with_capacity(cfg.depth);
    for i in 0..cfg.depth {
        let b = format!("{vp}.blocks.{i}");
        blocks.push(VisionBlock {
            norm1: layer_norm(ws, &format!("{b}.norm1"))?,
            norm2: layer_norm(ws, &format!("{b}.norm2"))?,
            attn: VisionAttention {
                qkv: linear(ws, &format!("{b}.attn.qkv"), true)?,
                proj: linear(ws, &format!("{b}.attn.proj"), true)?,
                num_heads: cfg.num_heads,
                head_dim,
            },
            mlp: VisionMlp {
                fc1: linear(ws, &format!("{b}.mlp.linear_fc1"), true)?,
                fc2: linear(ws, &format!("{b}.mlp.linear_fc2"), true)?,
            },
        });
    }

    let merger = patch_merger(ws, &format!("{vp}.merger"), merged_size, false)?;

    let mut deepstack_merger_list = Vec::with_capacity(cfg.deepstack_visual_indexes.len());
    for j in 0..cfg.deepstack_visual_indexes.len() {
        deepstack_merger_list.push(patch_merger(ws, &format!("{vp}.deepstack_merger_list.{j}"), merged_size, true)?);
    }

    Ok(VisionModel {
        patch_embed,
        pos_embed,
        rotary_emb,
        blocks,
        merger,
        deepstack_merger_list,
        deepstack_visual_indexes: cfg.deepstack_visual_indexes.clone(),
        spatial_merge_size: cfg.spatial_merge_size,
        num_grid_per_side,
        hidden_size: cfg.hidden_size,
    })
}

// ---------------------------------------------------------------------------
// Pool the last non-padding token from [batch, seq, hidden] hidden states.
// ---------------------------------------------------------------------------

pub(crate) fn pool_last(hidden: &Tensor, attention_mask: &Tensor) -> Tensor {
    let seq_len = attention_mask.size()[1];
    let last_pos = attention_mask.flip([1i64].as_ref()).argmax(1, false);
    let col = last_pos.neg() + (seq_len - 1);
    let batch = hidden.size()[0];
    let row = Tensor::arange(batch, (Kind::Int64, hidden.device()));
    hidden.index(&[Some(row), Some(col)])
}

// ---------------------------------------------------------------------------
// Dense: config, text model, score weight
// ---------------------------------------------------------------------------

async fn parse_dense_config(repo: &dyn ModelRepo) -> Result<(TextConfig, VisionConfig, i64)> {
    let path = repo.get_local_path("config.json").await?.context("config.json not found")?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let tc = &raw["text_config"];
    let vc = &raw["vision_config"];

    let num_attention_heads = tc["num_attention_heads"].as_i64().context("num_attention_heads")?;

    let mrope_section = tc["rope_scaling"]["mrope_section"]
        .as_array()
        .and_then(|a| if a.len() == 3 { Some([a[0].as_i64()?, a[1].as_i64()?, a[2].as_i64()?]) } else { None })
        .context("rope_scaling.mrope_section missing or malformed in config.json")?;

    let text_cfg = TextConfig {
        hidden_size: tc["hidden_size"].as_i64().context("hidden_size")?,
        intermediate_size: tc["intermediate_size"].as_i64().context("intermediate_size")?,
        num_hidden_layers: tc["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize,
        num_attention_heads,
        num_key_value_heads: tc["num_key_value_heads"].as_i64().unwrap_or(num_attention_heads),
        head_dim: tc["head_dim"].as_i64().context("head_dim")?,
        rms_norm_eps: tc["rms_norm_eps"].as_f64().context("rms_norm_eps")?,
        rope_theta: tc["rope_theta"].as_f64().unwrap_or(500_000.0),
        mrope_section,
    };

    let vision_cfg = parse_vision_config(vc, text_cfg.hidden_size)?;
    let image_token_id = raw["image_token_id"].as_i64().context("image_token_id missing from config.json")?;

    Ok((text_cfg, vision_cfg, image_token_id))
}

fn build_dense_text_model(ws: &WeightMap, cfg: &TextConfig) -> Result<TextModel> {
    let lm = "model.language_model";

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let l = format!("{lm}.layers.{i}");
        let sa = format!("{l}.self_attn");
        layers.push(DecoderLayer {
            attn: Attention {
                q_proj: linear(ws, &format!("{sa}.q_proj"), false)?,
                k_proj: linear(ws, &format!("{sa}.k_proj"), false)?,
                v_proj: linear(ws, &format!("{sa}.v_proj"), false)?,
                o_proj: linear(ws, &format!("{sa}.o_proj"), false)?,
                q_norm: rms_norm(ws, &format!("{sa}.q_norm"), cfg.rms_norm_eps)?,
                k_norm: rms_norm(ws, &format!("{sa}.k_norm"), cfg.rms_norm_eps)?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            },
            mlp: Mlp {
                gate_proj: linear(ws, &format!("{l}.mlp.gate_proj"), false)?,
                up_proj: linear(ws, &format!("{l}.mlp.up_proj"), false)?,
                down_proj: linear(ws, &format!("{l}.mlp.down_proj"), false)?,
            },
            input_norm: rms_norm(ws, &format!("{l}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attn_norm: rms_norm(ws, &format!("{l}.post_attention_layernorm"), cfg.rms_norm_eps)?,
        });
    }

    Ok(TextModel {
        embed_tokens: ws.get(&format!("{lm}.embed_tokens.weight"))?,
        layers,
        norm: rms_norm(ws, &format!("{lm}.norm"), cfg.rms_norm_eps)?,
        rotary_emb: RotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, cfg.mrope_section, ws.device),
    })
}

/// Builds the (yes − no) score vector from the lm_head for binary ranking.
fn build_score_weight(ws: &WeightMap, tokenizer: &tokenizers::Tokenizer) -> Result<Tensor> {
    let yes_id = tokenizer.token_to_id("yes").context("'yes' not in tokenizer vocab")? as i64;
    let no_id = tokenizer.token_to_id("no").context("'no' not in tokenizer vocab")? as i64;
    let lm_head = ws.get("lm_head.weight")?;
    Ok((lm_head.select(0, yes_id) - lm_head.select(0, no_id)).unsqueeze(0))
}

// ---------------------------------------------------------------------------
// MoE: config, text model
// ---------------------------------------------------------------------------

async fn parse_moe_config(repo: &dyn ModelRepo) -> Result<(MoeTextConfig, VisionConfig, i64)> {
    let path = repo.get_local_path("config.json").await?.context("config.json not found")?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let tc = &raw["text_config"];
    let vc = &raw["vision_config"];

    let num_attention_heads = tc["num_attention_heads"].as_i64().context("num_attention_heads")?;

    let num_experts = tc["num_local_experts"]
        .as_i64()
        .or_else(|| tc["num_experts"].as_i64())
        .context("num_local_experts / num_experts")?;

    let mlp_only_layers: Vec<usize> = tc["mlp_only_layers"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_default();

    let hidden_size = tc["hidden_size"].as_i64().context("hidden_size")?;

    let mrope_section = tc["rope_scaling"]["mrope_section"]
        .as_array()
        .and_then(|a| if a.len() == 3 { Some([a[0].as_i64()?, a[1].as_i64()?, a[2].as_i64()?]) } else { None })
        .context("rope_scaling.mrope_section missing or malformed in config.json")?;

    let text_cfg = MoeTextConfig {
        hidden_size,
        intermediate_size: tc["intermediate_size"].as_i64().context("intermediate_size")?,
        num_hidden_layers: tc["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize,
        num_attention_heads,
        num_key_value_heads: tc["num_key_value_heads"].as_i64().unwrap_or(num_attention_heads),
        head_dim: tc["head_dim"].as_i64().context("head_dim")?,
        rms_norm_eps: tc["rms_norm_eps"].as_f64().context("rms_norm_eps")?,
        rope_theta: tc["rope_theta"].as_f64().unwrap_or(500_000.0),
        mrope_section,
        decoder_sparse_step: tc["decoder_sparse_step"].as_u64().unwrap_or(1) as usize,
        moe_intermediate_size: tc["moe_intermediate_size"].as_i64().context("moe_intermediate_size")?,
        num_experts_per_tok: tc["num_experts_per_tok"].as_i64().context("num_experts_per_tok")?,
        num_experts,
        mlp_only_layers,
    };

    let vision_cfg = parse_vision_config(vc, hidden_size)?;
    let image_token_id = raw["image_token_id"].as_i64().context("image_token_id missing from config.json")?;

    Ok((text_cfg, vision_cfg, image_token_id))
}

fn build_moe_block(ws: &WeightMap, prefix: &str, cfg: &MoeTextConfig) -> Result<SparseMoeBlock> {
    Ok(SparseMoeBlock {
        gate: TopKRouter {
            weight: ws.get(&format!("{prefix}.gate.weight"))?,
            top_k: cfg.num_experts_per_tok,
        },
        experts: Experts {
            gate_up_proj: ws.get(&format!("{prefix}.experts.gate_up_proj"))?,
            down_proj: ws.get(&format!("{prefix}.experts.down_proj"))?,
        },
    })
}

fn build_moe_text_model(ws: &WeightMap, cfg: &MoeTextConfig) -> Result<MoeTextModel> {
    let lm = "model.language_model";

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let l = format!("{lm}.layers.{i}");
        let sa = format!("{l}.self_attn");

        let mlp = if is_moe_layer(i, cfg) {
            LayerMlp::Moe(build_moe_block(ws, &format!("{l}.mlp"), cfg)?)
        } else {
            LayerMlp::Dense(Mlp {
                gate_proj: linear(ws, &format!("{l}.mlp.gate_proj"), false)?,
                up_proj: linear(ws, &format!("{l}.mlp.up_proj"), false)?,
                down_proj: linear(ws, &format!("{l}.mlp.down_proj"), false)?,
            })
        };

        layers.push(MoeDecoderLayer {
            attn: Attention {
                q_proj: linear(ws, &format!("{sa}.q_proj"), false)?,
                k_proj: linear(ws, &format!("{sa}.k_proj"), false)?,
                v_proj: linear(ws, &format!("{sa}.v_proj"), false)?,
                o_proj: linear(ws, &format!("{sa}.o_proj"), false)?,
                q_norm: rms_norm(ws, &format!("{sa}.q_norm"), cfg.rms_norm_eps)?,
                k_norm: rms_norm(ws, &format!("{sa}.k_norm"), cfg.rms_norm_eps)?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
            },
            mlp,
            input_norm: rms_norm(ws, &format!("{l}.input_layernorm"), cfg.rms_norm_eps)?,
            post_attn_norm: rms_norm(ws, &format!("{l}.post_attention_layernorm"), cfg.rms_norm_eps)?,
        });
    }

    Ok(MoeTextModel {
        embed_tokens: ws.get(&format!("{lm}.embed_tokens.weight"))?,
        layers,
        norm: rms_norm(ws, &format!("{lm}.norm"), cfg.rms_norm_eps)?,
        rotary_emb: RotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, cfg.mrope_section, ws.device),
    })
}

// ---------------------------------------------------------------------------
// Shared vision config parser (eliminates duplication between dense/moe paths)
// ---------------------------------------------------------------------------

fn parse_vision_config(vc: &serde_json::Value, text_hidden_size: i64) -> Result<VisionConfig> {
    let deepstack_visual_indexes: Vec<usize> = vc["deepstack_visual_indexes"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
        .unwrap_or_else(|| vec![8, 16, 24]);

    Ok(VisionConfig {
        depth: vc["depth"].as_u64().unwrap_or(27) as usize,
        hidden_size: vc["hidden_size"].as_i64().unwrap_or(1152),
        intermediate_size: vc["intermediate_size"].as_i64().unwrap_or(4304),
        num_heads: vc["num_heads"].as_i64().unwrap_or(16),
        in_channels: vc["in_channels"].as_i64().unwrap_or(3),
        patch_size: vc["patch_size"].as_i64().unwrap_or(16),
        spatial_merge_size: vc["spatial_merge_size"].as_i64().unwrap_or(2),
        temporal_patch_size: vc["temporal_patch_size"].as_i64().unwrap_or(2),
        out_hidden_size: vc["out_hidden_size"].as_i64().unwrap_or(text_hidden_size),
        num_position_embeddings: vc["num_position_embeddings"].as_i64().unwrap_or(2304),
        deepstack_visual_indexes,
    })
}

// ---------------------------------------------------------------------------
// M-RoPE position ID builder for multimodal sequences
// ---------------------------------------------------------------------------

/// Builds 3D M-RoPE position IDs shaped [3, batch, seq].
///
/// Text tokens receive T = H = W = sequential offset.  Image token runs are
/// assigned 3D grid coordinates using their (T, H, W) patch grid divided by
/// `spatial_merge_size`; the next text position advances by max(llm_h, llm_w).
pub(crate) fn build_mrope_position_ids(input_ids: &Tensor, image_token_id: i64, grid_thw: &[(i64, i64, i64)], spatial_merge_size: i64) -> Tensor {
    let (batch, seq) = input_ids.size2().unwrap();
    let device = input_ids.device();
    let ms = spatial_merge_size;
    let flat_len = (3 * batch * seq) as usize;
    let mut buf = vec![0i64; flat_len];

    let ids_cpu = input_ids.to_device(tch::Device::Cpu).to_kind(tch::Kind::Int64);

    for b in 0..batch {
        let mut current_pos = 0i64;
        let mut img_idx = 0usize;
        let mut i = 0i64;

        while i < seq {
            if ids_cpu.int64_value(&[b, i]) == image_token_id && img_idx < grid_thw.len() {
                let (t, h, w) = grid_thw[img_idx];
                img_idx += 1;
                let llm_h = h / ms;
                let llm_w = w / ms;

                for ti in 0..t {
                    for hi in 0..llm_h {
                        for wi in 0..llm_w {
                            let tok = i + ti * llm_h * llm_w + hi * llm_w + wi;
                            let base = b * seq + tok;
                            buf[(0 * batch * seq + base) as usize] = current_pos + ti;
                            buf[(1 * batch * seq + base) as usize] = current_pos + hi;
                            buf[(2 * batch * seq + base) as usize] = current_pos + wi;
                        }
                    }
                }

                current_pos += llm_h.max(llm_w);
                i += t * llm_h * llm_w;
            } else {
                let base = b * seq + i;
                buf[(0 * batch * seq + base) as usize] = current_pos;
                buf[(1 * batch * seq + base) as usize] = current_pos;
                buf[(2 * batch * seq + base) as usize] = current_pos;
                current_pos += 1;
                i += 1;
            }
        }
    }

    Tensor::from_slice(&buf).reshape([3, batch, seq]).to_device(device)
}

// ---------------------------------------------------------------------------
// Tokenized data for Qwen3-VL (text-only or multimodal)
// ---------------------------------------------------------------------------

/// Output of the Qwen3-VL tokenizer. Carries token IDs plus optional vision
/// tensors so a single `embed` call covers both text-only and multimodal inputs.
pub struct Qwen3VLTokenizedData {
    pub input_ids: Tensor,
    pub pixel_values: Option<Tensor>,
    pub grid_thw: Option<Vec<(i64, i64, i64)>>,
}

impl TokenizedData for Qwen3VLTokenizedData {}

// ---------------------------------------------------------------------------
// Shared text-forward trait (bridges TextModel and MoeTextModel)
// ---------------------------------------------------------------------------

trait TextForward {
    fn fwd(&self, input_ids: &Tensor) -> Tensor;
    fn fwd_mm(&self, input_ids: &Tensor, pos: &Tensor, img: &Tensor, ds: &[Tensor], tok: i64) -> Tensor;
}

impl TextForward for TextModel {
    fn fwd(&self, input_ids: &Tensor) -> Tensor {
        self.forward(input_ids)
    }
    fn fwd_mm(&self, input_ids: &Tensor, pos: &Tensor, img: &Tensor, ds: &[Tensor], tok: i64) -> Tensor {
        self.forward_multimodal(input_ids, pos, img, ds, tok)
    }
}

impl TextForward for MoeTextModel {
    fn fwd(&self, input_ids: &Tensor) -> Tensor {
        self.forward(input_ids)
    }
    fn fwd_mm(&self, input_ids: &Tensor, pos: &Tensor, img: &Tensor, ds: &[Tensor], tok: i64) -> Tensor {
        self.forward_multimodal(input_ids, pos, img, ds, tok)
    }
}

fn embed_inner(
    model: &dyn TextForward, vision: Option<&Arc<VisionModel>>, pad_token_id: i64, image_token_id: i64, spatial_merge_size: i64, info: &dyn TokenizedData,
) -> Result<Tensor> {
    let data = (info as &dyn std::any::Any)
        .downcast_ref::<Qwen3VLTokenizedData>()
        .ok_or_else(|| anyhow::anyhow!("TokenizedData is not Qwen3VLTokenizedData"))?;
    let input_ids = &data.input_ids;
    let vision_out = match (data.pixel_values.as_ref(), data.grid_thw.as_deref(), vision) {
        (Some(pv), Some(thw), Some(vis)) => Some(tch::no_grad(|| vis.forward(pv, thw))),
        _ => None,
    };
    let hidden = match (&vision_out, data.grid_thw.as_deref()) {
        (Some(vo), Some(thw)) => {
            let pos_ids = build_mrope_position_ids(input_ids, image_token_id, thw, spatial_merge_size);
            tch::no_grad(|| model.fwd_mm(input_ids, &pos_ids, &vo.image_features, &vo.deepstack_features, image_token_id))
        }
        _ => tch::no_grad(|| model.fwd(input_ids)),
    };
    let mask = input_ids.ne(pad_token_id).to_kind(Kind::Int64);
    let emb = pool_last(&hidden, &mask);
    let norm = emb.norm_scalaropt_dim(2.0_f64, [-1i64].as_ref(), true).clamp_min(1.0e-12);
    Ok(emb / norm)
}

// ---------------------------------------------------------------------------
// Dense embedding / ranking model impls
// ---------------------------------------------------------------------------

struct EmbeddingModelImpl {
    model: Arc<TextModel>,
    vision: Option<Arc<VisionModel>>,
    pad_token_id: i64,
    image_token_id: i64,
    spatial_merge_size: i64,
}

impl EmbeddingModel for EmbeddingModelImpl {
    fn embed(&self, info: &dyn TokenizedData) -> Result<Tensor> {
        embed_inner(&*self.model, self.vision.as_ref(), self.pad_token_id, self.image_token_id, self.spatial_merge_size, info)
    }
}

struct RankingModelImpl {
    model: Arc<TextModel>,
    score_linear: Linear,
}

impl RankingModel for RankingModelImpl {
    fn rank(&self, docs: &[Tensor]) -> Result<Tensor> {
        let scores: Vec<Tensor> = docs
            .iter()
            .map(|doc| {
                let hidden = tch::no_grad(|| self.model.forward(doc));
                let last = hidden.select(1, -1);
                self.score_linear.forward(&last).sigmoid().squeeze_dim(-1)
            })
            .collect();
        Ok(Tensor::cat(&scores, 0))
    }
}

struct MoeEmbeddingModelImpl {
    model: Arc<MoeTextModel>,
    vision: Option<Arc<VisionModel>>,
    pad_token_id: i64,
    image_token_id: i64,
    spatial_merge_size: i64,
}

impl EmbeddingModel for MoeEmbeddingModelImpl {
    fn embed(&self, info: &dyn TokenizedData) -> Result<Tensor> {
        embed_inner(&*self.model, self.vision.as_ref(), self.pad_token_id, self.image_token_id, self.spatial_merge_size, info)
    }
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

/// Resizes, normalizes, and patches a still image for the Qwen3-VL vision encoder.
/// Returns pixel_values `[T*H*W, C*tp*P*P]` and the grid `(T, H, W)` in patch units.
pub(crate) fn preprocess_image(
    img: &image::DynamicImage, patch_size: i64, spatial_merge_size: i64, temporal_patch_size: i64, in_channels: i64,
) -> Result<(Tensor, (i64, i64, i64))> {
    let tile = patch_size * spatial_merge_size;
    let (ow, oh) = (img.width() as i64, img.height() as i64);
    let new_h = ((oh as f64 / tile as f64).round() as i64 * tile).max(tile);
    let new_w = ((ow as f64 / tile as f64).round() as i64 * tile).max(tile);
    let rgb = img.resize_exact(new_w as u32, new_h as u32, image::imageops::FilterType::CatmullRom).into_rgb8();
    let raw = rgb.as_raw();
    let (h, w, c) = (new_h as usize, new_w as usize, in_channels as usize);
    let mut chw = vec![0.0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ci in 0..c {
                chw[ci * h * w + y * w + x] = (raw[(y * w + x) * c + ci] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    let (gh, gw, tp, p, ms) = (new_h / patch_size, new_w / patch_size, temporal_patch_size, patch_size, spatial_merge_size);
    let frame = Tensor::from_slice(&chw).reshape([in_channels, new_h, new_w]).to_kind(Kind::BFloat16);
    let frames = frame.unsqueeze(0).expand([tp, -1, -1, -1], false).contiguous().unsqueeze(0);
    let r = frames
        .reshape([1, 1, tp, in_channels, gh / ms, ms, p, gw / ms, ms, p])
        .permute([0, 1, 4, 7, 5, 8, 3, 2, 6, 9]);
    Ok((r.reshape([gh * gw, in_channels * tp * p * p]), (1, gh, gw)))
}

// ---------------------------------------------------------------------------
// Image tokenizer
// ---------------------------------------------------------------------------

struct Qwen3VLImageTokenizer {
    patch_size: i64,
    spatial_merge_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,
    image_token_id: i64,
}

impl ImageTokenizer for Qwen3VLImageTokenizer {
    fn encode(&self, img: &image::DynamicImage) -> anyhow::Result<Box<dyn TokenizedData>> {
        let (pixel_values, (t, h, w)) = preprocess_image(img, self.patch_size, self.spatial_merge_size, self.temporal_patch_size, self.in_channels)?;
        let num_tokens = t * (h / self.spatial_merge_size) * (w / self.spatial_merge_size);
        let input_ids = Tensor::full([1, num_tokens], self.image_token_id, (Kind::Int64, tch::Device::Cpu));
        Ok(Box::new(Qwen3VLTokenizedData {
            input_ids,
            pixel_values: Some(pixel_values),
            grid_thw: Some(vec![(t, h, w)]),
        }))
    }
}

// ---------------------------------------------------------------------------
// Text tokenizer
// ---------------------------------------------------------------------------

struct Qwen3VLTextTokenizer {
    inner: tokenizers::Tokenizer,
}

impl TextTokenizer for Qwen3VLTextTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Box<dyn TokenizedData>> {
        let enc = self.inner.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let ids = enc.get_ids();
        let seq = ids.len() as i64;
        anyhow::ensure!(ids.iter().all(|&x| x < i32::MIN as u32), "token id exceeds i32 domain");
        // SAFETY: all ids have high bit clear so u32 and i32 bit patterns are identical.
        let input_ids = unsafe {
            let i32s = std::slice::from_raw_parts(ids.as_ptr() as *const i32, ids.len());
            Tensor::from_slice(i32s).to_kind(Kind::Int64).reshape([1, seq])
        };
        Ok(Box::new(Qwen3VLTokenizedData {
            input_ids,
            pixel_values: None,
            grid_thw: None,
        }))
    }
    fn decode(&self, tokens: Tensor) -> anyhow::Result<String> {
        let ids: Vec<i64> = tokens.flatten(0, -1).try_into()?;
        self.inner
            .decode(&ids.iter().map(|&x| x as u32).collect::<Vec<_>>(), true)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ---------------------------------------------------------------------------
// ModelLoader implementations
// ---------------------------------------------------------------------------

struct Qwen3VLDenseModelLoader {
    id: ModelIds,
    text_cfg: TextConfig,
    vision_cfg: VisionConfig,
    image_token_id: i64,
    tokenizer: tokenizers::Tokenizer,
    shard_paths: Vec<PathBuf>,
}

impl ModelLoader for Qwen3VLDenseModelLoader {
    fn identifier(&self) -> Option<ModelIds> {
        Some(self.id)
    }
    fn initialize(&self, device: tch::Device) -> Result<Box<dyn LocalModelBuilder>> {
        let ws = WeightMap::from_paths(self.shard_paths.clone(), device)?;
        let pad = self.tokenizer.token_to_id("<|endoftext|>").context("pad token missing")? as i64;
        let sw = matches!(self.id, ModelIds::Qwen3VLReranker).then(|| build_score_weight(&ws, &self.tokenizer)).transpose()?;
        Ok(Box::new(Qwen3VLBuilder {
            id: self.id,
            model: Arc::new(build_dense_text_model(&ws, &self.text_cfg)?),
            vision: Some(Arc::new(build_vision_model(&ws, &self.vision_cfg)?)),
            score_weight: sw,
            pad_token_id: pad,
            image_token_id: self.image_token_id,
            spatial_merge_size: self.vision_cfg.spatial_merge_size,
            patch_size: self.vision_cfg.patch_size,
            temporal_patch_size: self.vision_cfg.temporal_patch_size,
            in_channels: self.vision_cfg.in_channels,
            tokenizer: self.tokenizer.clone(),
        }))
    }
}

struct Qwen3VLMoeModelLoader {
    text_cfg: MoeTextConfig,
    vision_cfg: VisionConfig,
    image_token_id: i64,
    tokenizer: tokenizers::Tokenizer,
    shard_paths: Vec<PathBuf>,
}

impl ModelLoader for Qwen3VLMoeModelLoader {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLMoe)
    }
    fn initialize(&self, device: tch::Device) -> Result<Box<dyn LocalModelBuilder>> {
        let ws = WeightMap::from_paths(self.shard_paths.clone(), device)?;
        let pad = self.tokenizer.token_to_id("<|endoftext|>").context("pad token missing")? as i64;
        Ok(Box::new(Qwen3VLMoeBuilder {
            model: Arc::new(build_moe_text_model(&ws, &self.text_cfg)?),
            vision: Some(Arc::new(build_vision_model(&ws, &self.vision_cfg)?)),
            pad_token_id: pad,
            image_token_id: self.image_token_id,
            spatial_merge_size: self.vision_cfg.spatial_merge_size,
            patch_size: self.vision_cfg.patch_size,
            temporal_patch_size: self.vision_cfg.temporal_patch_size,
            in_channels: self.vision_cfg.in_channels,
            tokenizer: self.tokenizer.clone(),
        }))
    }
}

// ---------------------------------------------------------------------------
// Dense builder
// ---------------------------------------------------------------------------

/// Holds a loaded Qwen3-VL text model and produces embedding or ranking runners.
pub struct Qwen3VLBuilder {
    id: ModelIds,
    model: Arc<TextModel>,
    vision: Option<Arc<VisionModel>>,
    score_weight: Option<Tensor>,
    pad_token_id: i64,
    image_token_id: i64,
    spatial_merge_size: i64,
    patch_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,
    tokenizer: tokenizers::Tokenizer,
}

impl LocalModelBuilder for Qwen3VLBuilder {
    fn identifier(&self) -> Option<ModelIds> {
        Some(self.id)
    }

    fn text_tokenizer(&self) -> Option<Box<dyn TextTokenizer>> {
        Some(Box::new(Qwen3VLTextTokenizer { inner: self.tokenizer.clone() }))
    }

    fn image_tokenizer(&self) -> Option<Box<dyn ImageTokenizer>> {
        Some(Box::new(Qwen3VLImageTokenizer {
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
            image_token_id: self.image_token_id,
        }))
    }

    fn is_embedding_model(&self) -> bool {
        matches!(self.id, ModelIds::Qwen3VLEmbedding)
    }

    fn get_embedding_model(&self, _scheme: EmbeddingScheme) -> Option<Result<Box<dyn EmbeddingModel>>> {
        if !self.is_embedding_model() {
            return None;
        }
        Some(Ok(Box::new(EmbeddingModelImpl {
            model: Arc::clone(&self.model),
            vision: self.vision.as_ref().map(Arc::clone),
            pad_token_id: self.pad_token_id,
            image_token_id: self.image_token_id,
            spatial_merge_size: self.spatial_merge_size,
        })))
    }

    fn is_ranking_model(&self) -> bool {
        matches!(self.id, ModelIds::Qwen3VLReranker)
    }

    fn get_ranking_model(&self, _scheme: EmbeddingScheme) -> Option<Result<Box<dyn RankingModel>>> {
        if !self.is_ranking_model() {
            return None;
        }
        let weight = self.score_weight.as_ref()?.shallow_clone();
        Some(Ok(Box::new(RankingModelImpl {
            model: Arc::clone(&self.model),
            score_linear: Linear { weight, bias: None },
        })))
    }
}

// ---------------------------------------------------------------------------
// MoE builder
// ---------------------------------------------------------------------------

/// Holds a loaded Qwen3-VL-MoE text model and produces an embedding runner.
pub struct Qwen3VLMoeBuilder {
    model: Arc<MoeTextModel>,
    vision: Option<Arc<VisionModel>>,
    pad_token_id: i64,
    image_token_id: i64,
    spatial_merge_size: i64,
    patch_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,
    tokenizer: tokenizers::Tokenizer,
}

impl LocalModelBuilder for Qwen3VLMoeBuilder {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLMoe)
    }

    fn text_tokenizer(&self) -> Option<Box<dyn TextTokenizer>> {
        Some(Box::new(Qwen3VLTextTokenizer { inner: self.tokenizer.clone() }))
    }

    fn image_tokenizer(&self) -> Option<Box<dyn ImageTokenizer>> {
        Some(Box::new(Qwen3VLImageTokenizer {
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
            temporal_patch_size: self.temporal_patch_size,
            in_channels: self.in_channels,
            image_token_id: self.image_token_id,
        }))
    }

    fn is_embedding_model(&self) -> bool {
        true
    }

    fn get_embedding_model(&self, _scheme: EmbeddingScheme) -> Option<Result<Box<dyn EmbeddingModel>>> {
        Some(Ok(Box::new(MoeEmbeddingModelImpl {
            model: Arc::clone(&self.model),
            vision: self.vision.as_ref().map(Arc::clone),
            pad_token_id: self.pad_token_id,
            image_token_id: self.image_token_id,
            spatial_merge_size: self.spatial_merge_size,
        })))
    }

    fn is_ranking_model(&self) -> bool {
        false
    }

    fn get_ranking_model(&self, _scheme: EmbeddingScheme) -> Option<Result<Box<dyn RankingModel>>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Load logic
// ---------------------------------------------------------------------------

async fn load_dense_builder(repo: Box<dyn ModelRepo>, id: ModelIds) -> Result<Box<dyn ModelLoader>> {
    let (text_cfg, vision_cfg, image_token_id) = parse_dense_config(repo.as_ref()).await?;
    let tokenizer_path = repo.get_local_path("tokenizer.json").await?.context("tokenizer.json not found")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
    let shard_paths = resolve_shard_paths(repo.as_ref()).await?;
    Ok(Box::new(Qwen3VLDenseModelLoader {
        id,
        text_cfg,
        vision_cfg,
        image_token_id,
        tokenizer,
        shard_paths,
    }))
}

async fn load_moe_builder(repo: Box<dyn ModelRepo>) -> Result<Box<dyn ModelLoader>> {
    let (text_cfg, vision_cfg, image_token_id) = parse_moe_config(repo.as_ref()).await?;
    let tokenizer_path = repo.get_local_path("tokenizer.json").await?.context("tokenizer.json not found")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;
    let shard_paths = resolve_shard_paths(repo.as_ref()).await?;
    Ok(Box::new(Qwen3VLMoeModelLoader {
        text_cfg,
        vision_cfg,
        image_token_id,
        tokenizer,
        shard_paths,
    }))
}

// ---------------------------------------------------------------------------
// ModelFactory implementations
// ---------------------------------------------------------------------------

/// Factory for the embedding variant.
pub struct Qwen3VLEmbeddingFactory;

impl ModelFactory for Qwen3VLEmbeddingFactory {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLEmbedding)
    }
    fn load<'a>(&'a self, repo: Box<dyn ModelRepo>) -> Pin<Box<dyn Future<Output = Result<Box<dyn ModelLoader>>> + 'a>> {
        Box::pin(load_dense_builder(repo, self.identifier().unwrap()))
    }
}

/// Factory for the reranker variant.
pub struct Qwen3VLRerankerFactory;

impl ModelFactory for Qwen3VLRerankerFactory {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLReranker)
    }
    fn load<'a>(&'a self, repo: Box<dyn ModelRepo>) -> Pin<Box<dyn Future<Output = Result<Box<dyn ModelLoader>>> + 'a>> {
        Box::pin(load_dense_builder(repo, self.identifier().unwrap()))
    }
}

/// Factory for the Qwen3-VL-MoE embedding variant.
pub struct Qwen3VLMoeFactory;

impl ModelFactory for Qwen3VLMoeFactory {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLMoe)
    }
    fn load<'a>(&'a self, repo: Box<dyn ModelRepo>) -> Pin<Box<dyn Future<Output = Result<Box<dyn ModelLoader>>> + 'a>> {
        Box::pin(load_moe_builder(repo))
    }
}

#[cfg(test)]
/// Loads the dense `TextModel` from a local directory for use in tests.
pub(crate) async fn load_text_model_from_dir(p: &std::path::Path) -> Result<dense::TextModel> {
    let repo = init_repo(p).await?;
    let (cfg, _, _) = parse_dense_config(repo.as_ref()).await?;
    build_dense_text_model(&WeightMap::load(repo.as_ref(), tch::Device::Cpu).await?, &cfg)
}

#[cfg(test)]
/// Loads the dense text + vision models for step-by-step debug tests.
pub(crate) async fn load_dense_for_debug(p: &std::path::Path, device: tch::Device) -> Result<(dense::TextModel, VisionModel, vision::VisionConfig, i64)> {
    let repo = init_repo(p).await?;
    let (tcfg, vcfg, image_token_id) = parse_dense_config(repo.as_ref()).await?;
    let ws = WeightMap::load(repo.as_ref(), device).await?;
    Ok((build_dense_text_model(&ws, &tcfg)?, build_vision_model(&ws, &vcfg)?, vcfg, image_token_id))
}
