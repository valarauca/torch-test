mod dense;

use std::{collections::HashSet, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use tch::{Kind, Tensor};

use crate::tensors::SafeTensor;
use crate::transformers::{
    model_ids::ModelIds,
    traits::{
        EmbeddingModel, EmbeddingScheme, ImageTokenizer, LocalModelBuilder, ModelFactory,
        PinnedFuture, RankingModel, TextTokenizer,
    },
};
use dense::{Attention, DecoderLayer, Linear, Mlp, RotaryEmbedding, RmsNorm, TextConfig, TextModel};

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
    fn load(repo: &dyn crate::transformers::traits::ModelRepo, device: tch::Device) -> Result<Self> {
        let mut shards = Vec::new();

        if let Some(path) = repo.get_local_path("model.safetensors")? {
            shards.push(SafeTensor::load(&path)?);
        } else {
            let idx_path = repo
                .get_local_path("model.safetensors.index.json")?
                .context("neither model.safetensors nor model.safetensors.index.json found")?;

            let idx_str = std::fs::read_to_string(&idx_path)?;
            let idx: serde_json::Value = serde_json::from_str(&idx_str)?;
            let weight_map = idx["weight_map"].as_object().context("invalid index: missing weight_map")?;

            let shard_names: HashSet<&str> = weight_map.values()
                .filter_map(|v| v.as_str())
                .collect();

            for shard_name in shard_names {
                let path = repo.get_local_path(shard_name)?
                    .with_context(|| format!("shard not found: {shard_name}"))?;
                shards.push(SafeTensor::load(&path)?);
            }
        }

        Ok(Self { shards, device })
    }

    /// Finds `name` across all shards and returns an independent copy on the target device.
    fn get(&self, name: &str) -> Result<Tensor> {
        for shard in &self.shards {
            if let Some(kind) = shard.kind_of(name) {
                let wrapper = shard
                    .get_tensor(name, kind, tch::Device::Cpu)?
                    .unwrap();
                let mut out = Tensor::empty(wrapper.size().as_slice(), (kind, self.device));
                out.copy_(&*wrapper);
                return Ok(out);
            }
        }
        anyhow::bail!("weight not found: {name}")
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

fn parse_config(repo: &dyn crate::transformers::traits::ModelRepo) -> Result<TextConfig> {
    let path = repo
        .get_local_path("config.json")?
        .context("config.json not found")?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let tc = &raw["text_config"];

    let num_attention_heads = tc["num_attention_heads"].as_i64().context("num_attention_heads")?;

    Ok(TextConfig {
        hidden_size: tc["hidden_size"].as_i64().context("hidden_size")?,
        intermediate_size: tc["intermediate_size"].as_i64().context("intermediate_size")?,
        num_hidden_layers: tc["num_hidden_layers"].as_u64().context("num_hidden_layers")? as usize,
        num_attention_heads,
        num_key_value_heads: tc["num_key_value_heads"].as_i64().unwrap_or(num_attention_heads),
        head_dim: tc["head_dim"].as_i64().context("head_dim")?,
        rms_norm_eps: tc["rms_norm_eps"].as_f64().context("rms_norm_eps")?,
        rope_theta: tc["rope_parameters"]["rope_theta"].as_f64().unwrap_or(500_000.0),
    })
}

// ---------------------------------------------------------------------------
// Model construction helpers
// ---------------------------------------------------------------------------

fn linear(ws: &WeightMap, prefix: &str, has_bias: bool) -> Result<Linear> {
    Ok(Linear {
        weight: ws.get(&format!("{prefix}.weight"))?,
        bias: if has_bias { Some(ws.get(&format!("{prefix}.bias"))?) } else { None },
    })
}

fn rms_norm(ws: &WeightMap, prefix: &str, eps: f64) -> Result<RmsNorm> {
    Ok(RmsNorm { weight: ws.get(&format!("{prefix}.weight"))?, eps })
}

fn build_text_model(ws: &WeightMap, cfg: &TextConfig) -> Result<TextModel> {
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
                up_proj:   linear(ws, &format!("{l}.mlp.up_proj"),   false)?,
                down_proj: linear(ws, &format!("{l}.mlp.down_proj"), false)?,
            },
            input_norm:    rms_norm(ws, &format!("{l}.input_layernorm"),          cfg.rms_norm_eps)?,
            post_attn_norm: rms_norm(ws, &format!("{l}.post_attention_layernorm"), cfg.rms_norm_eps)?,
        });
    }

    Ok(TextModel {
        embed_tokens: ws.get(&format!("{lm}.embed_tokens.weight"))?,
        layers,
        norm: rms_norm(ws, &format!("{lm}.norm"), cfg.rms_norm_eps)?,
        rotary_emb: RotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, ws.device),
    })
}

/// Builds the (yes − no) score vector from the lm_head for binary ranking.
fn build_score_weight(
    ws: &WeightMap,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Tensor> {
    let yes_id = tokenizer.token_to_id("yes").context("'yes' not in tokenizer vocab")? as i64;
    let no_id  = tokenizer.token_to_id("no").context("'no' not in tokenizer vocab")? as i64;
    let lm_head = ws.get("lm_head.weight")?;
    Ok((lm_head.select(0, yes_id) - lm_head.select(0, no_id)).unsqueeze(0))
}

// ---------------------------------------------------------------------------
// Pool the last non-padding token from [batch, seq, hidden] hidden states.
// ---------------------------------------------------------------------------

fn pool_last(hidden: &Tensor, attention_mask: &Tensor) -> Tensor {
    let seq_len = attention_mask.size()[1];
    // Find the index of the last 1 in each row by flipping and taking argmax.
    let last_pos = attention_mask.flip([1i64].as_ref()).argmax(1, false);
    let col = last_pos.neg() + (seq_len - 1);
    let batch = hidden.size()[0];
    let row = Tensor::arange(batch, (Kind::Int64, hidden.device()));
    hidden.index(&[Some(row), Some(col)])
}

// ---------------------------------------------------------------------------
// Embedding model
// ---------------------------------------------------------------------------

struct EmbeddingModelImpl {
    model: Arc<TextModel>,
    pad_token_id: i64,
}

impl EmbeddingModel for EmbeddingModelImpl {
    fn embed(&self, input_ids: Tensor) -> Result<Tensor> {
        let model = &self.model;
        let hidden = tch::no_grad(|| model.forward(&input_ids));
        let mask = input_ids.ne(self.pad_token_id).to_kind(Kind::Int64);
        let emb = pool_last(&hidden, &mask);
        let norm = emb.norm_scalaropt_dim(2.0_f64, [-1i64].as_ref(), true).clamp_min(1e-12);
        Ok(emb / norm)
    }
}

// ---------------------------------------------------------------------------
// Ranking model
// ---------------------------------------------------------------------------

struct RankingModelImpl {
    model: Arc<TextModel>,
    /// (yes_weight − no_weight) from lm_head, no bias.
    score_linear: Linear,
}

impl RankingModel for RankingModelImpl {
    fn rank(&self, docs: &[Tensor]) -> Result<Tensor> {
        let model = &self.model;
        let scores: Vec<Tensor> = docs.iter().map(|doc| {
            let hidden = tch::no_grad(|| model.forward(doc));
            let last = hidden.select(1, -1);
            self.score_linear.forward(&last).sigmoid().squeeze_dim(-1)
        }).collect();
        Ok(Tensor::cat(&scores, 0))
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Holds a loaded Qwen3-VL text model and produces embedding or ranking runners.
pub struct Qwen3VLBuilder {
    id: ModelIds,
    model: Arc<TextModel>,
    score_weight: Option<Tensor>,
    pad_token_id: i64,
}

impl LocalModelBuilder for Qwen3VLBuilder {
    fn identifier(&self) -> Option<ModelIds> {
        Some(self.id)
    }

    fn text_tokenizer(&self) -> Option<Box<dyn TextTokenizer>> {
        None
    }

    fn image_tokenizer(&self) -> Option<Box<dyn ImageTokenizer>> {
        None
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
            pad_token_id: self.pad_token_id,
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
// Shared load logic
// ---------------------------------------------------------------------------

async fn load_builder(
    repo: Box<dyn crate::transformers::traits::ModelRepo>,
    device: tch::Device,
    id: ModelIds,
) -> Result<Box<dyn LocalModelBuilder>> {
    let cfg = parse_config(repo.as_ref())?;
    let ws  = WeightMap::load(repo.as_ref(), device)?;

    let tokenizer_path: PathBuf = repo
        .get_local_path("tokenizer.json")?
        .context("tokenizer.json not found")?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load error: {e}"))?;

    let pad_token_id = tokenizer
        .token_to_id("<|endoftext|>")
        .unwrap_or(0) as i64;

    let score_weight = if matches!(id, ModelIds::Qwen3VLReranker) {
        Some(build_score_weight(&ws, &tokenizer)?)
    } else {
        None
    };

    let model = Arc::new(build_text_model(&ws, &cfg)?);

    Ok(Box::new(Qwen3VLBuilder { id, model, score_weight, pad_token_id }))
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

    fn load(repo: Box<dyn crate::transformers::traits::ModelRepo>, device: tch::Device) -> PinnedFuture<Box<dyn LocalModelBuilder>> {
        Box::pin(load_builder(repo, device, ModelIds::Qwen3VLEmbedding))
    }
}

/// Factory for the reranker variant.
pub struct Qwen3VLRerankerFactory;

impl ModelFactory for Qwen3VLRerankerFactory {
    fn identifier(&self) -> Option<ModelIds> {
        Some(ModelIds::Qwen3VLReranker)
    }

    fn load(repo: Box<dyn crate::transformers::traits::ModelRepo>, device: tch::Device) -> PinnedFuture<Box<dyn LocalModelBuilder>> {
        Box::pin(load_builder(repo, device, ModelIds::Qwen3VLReranker))
    }
}
