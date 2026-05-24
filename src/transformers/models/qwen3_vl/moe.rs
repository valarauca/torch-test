use tch::{Kind, Tensor};

use super::dense::{Attention, Mlp, RmsNorm, RotaryEmbedding};

/// Hyperparameters for the Qwen3-VL-MoE text transformer.
#[derive(Clone, Debug)]
pub struct MoeTextConfig {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Number of RoPE frequency components assigned to [temporal, height, width].
    /// Must sum to head_dim / 2.  Default [24, 20, 20] is correct for head_dim = 128.
    pub mrope_section: [i64; 3],
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: i64,
    pub num_experts_per_tok: i64,
    pub num_experts: i64,
    pub mlp_only_layers: Vec<usize>,
}

/// Returns true when layer `i` uses a SparseMoeBlock rather than a dense Mlp.
pub fn is_moe_layer(i: usize, cfg: &MoeTextConfig) -> bool {
    !cfg.mlp_only_layers.contains(&i)
        && cfg.num_experts > 0
        && (i + 1) % cfg.decoder_sparse_step == 0
}

/// All expert weights stacked into 3-D tensors:
///   gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
///   down_proj:    [num_experts, hidden_size, moe_intermediate_size]
pub struct Experts {
    pub gate_up_proj: Tensor,
    pub down_proj: Tensor,
}

impl Experts {
    /// Dispatch tokens to their assigned experts and accumulate weighted outputs.
    /// hidden_states: [num_tokens, hidden_size]
    /// top_k_index:   [num_tokens, top_k]  (Int64)
    /// top_k_weights: [num_tokens, top_k]  (same dtype as hidden_states)
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        top_k_index: &Tensor,
        top_k_weights: &Tensor,
    ) -> Tensor {
        let num_experts = self.gate_up_proj.size()[0];
        let d_e = self.gate_up_proj.size()[1] / 2;
        let device = hidden_states.device();
        let mut output = Tensor::zeros_like(hidden_states);

        let expert_mask = top_k_index.one_hot(num_experts).permute([2i64, 1, 0].as_ref());

        let expert_hit = expert_mask
            .sum_dim_intlist([1i64, 2].as_ref(), false, Kind::Int64)
            .gt(0)
            .nonzero();

        for i in 0..expert_hit.size()[0] {
            let expert_idx = expert_hit.int64_value(&[i, 0]);

            let nz = expert_mask.select(0, expert_idx).nonzero();
            if nz.size()[0] == 0 {
                continue;
            }
            let top_k_pos = nz.select(1, 0);
            let token_idx = nz.select(1, 1);

            let current = hidden_states.index_select(0, &token_idx);

            let gate_up =
                current.linear(&self.gate_up_proj.select(0, expert_idx), None::<&Tensor>);
            let gate = gate_up.narrow(1, 0, d_e);
            let up = gate_up.narrow(1, d_e, d_e);
            let h = gate.silu() * up;

            let out = h.linear(&self.down_proj.select(0, expert_idx), None::<&Tensor>);

            let n = token_idx.size()[0];
            let row = Tensor::arange(n, (Kind::Int64, device));
            let w = top_k_weights
                .index_select(0, &token_idx)
                .index(&[Some(row), Some(top_k_pos)])
                .unsqueeze(-1);

            let _ = output.index_add_(0, &token_idx, &(out * w).to_kind(output.kind()));
        }

        output
    }
}

/// Softmax top-k router.  weight: [num_experts, hidden_size].
pub struct TopKRouter {
    pub weight: Tensor,
    pub top_k: i64,
}

impl TopKRouter {
    /// Returns (normalized_weights, expert_indices) both shaped [num_tokens, top_k].
    pub fn forward(&self, hidden_states: &Tensor) -> (Tensor, Tensor) {
        let logits = hidden_states.linear(&self.weight, None::<&Tensor>);
        let probs = logits.softmax(-1, Kind::Float);
        let (mut top_values, top_indices) = probs.topk(self.top_k, -1, true, true);
        let sum = top_values.sum_dim_intlist([-1i64].as_ref(), true, Kind::Float);
        top_values /= &sum;
        (top_values.to_kind(logits.kind()), top_indices)
    }
}

/// Sparse MoE block: routes each token to top-k experts, accumulates results.
pub struct SparseMoeBlock {
    pub experts: Experts,
    pub gate: TopKRouter,
}

impl SparseMoeBlock {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, seq, hidden) = x.size3().unwrap();
        let flat = x.reshape([-1, hidden]);
        let (weights, indices) = self.gate.forward(&flat);
        let out = self.experts.forward(&flat, &indices, &weights);
        out.reshape([batch, seq, hidden])
    }
}

/// Per-layer FFN: either the dense SwiGLU MLP or a sparse MoE block.
pub enum LayerMlp {
    Dense(Mlp),
    Moe(SparseMoeBlock),
}

/// Pre-norm decoder layer identical to the dense variant except the FFN may be sparse.
pub struct MoeDecoderLayer {
    pub attn: Attention,
    pub mlp: LayerMlp,
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
}

impl MoeDecoderLayer {
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let attn_out = self.attn.forward(&self.input_norm.forward(x), cos, sin);
        let h = x + &attn_out;
        let normed = self.post_attn_norm.forward(&h);
        let mlp_out = match &self.mlp {
            LayerMlp::Dense(mlp) => mlp.forward(&normed),
            LayerMlp::Moe(moe) => moe.forward(&normed),
        };
        &h + &mlp_out
    }
}

/// Full Qwen3-VL-MoE text transformer (text-only path).
pub struct MoeTextModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<MoeDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: RotaryEmbedding,
}

impl MoeTextModel {
    /// Returns final hidden states shaped [batch, seq, hidden_size].
    ///
    /// Builds text-only M-RoPE position IDs (T = H = W = token offset).
    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let (batch, seq) = input_ids.size2().unwrap();
        let device = input_ids.device();

        let pos_1d = Tensor::arange(seq, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([batch, -1], false);
        let position_ids = pos_1d.unsqueeze(0).expand([3, -1, -1], false);
        let (cos, sin) = self.rotary_emb.forward(&position_ids);

        let flat = input_ids.reshape([-1]);
        let mut h = self.embed_tokens.index_select(0, &flat).reshape([batch, seq, -1]);

        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin);
        }

        self.norm.forward(&h)
    }

    /// Multimodal forward: replaces image-token embeddings with vision features and
    /// applies per-layer DeepStack injection for the first `deepstack_features.len()` layers.
    ///
    /// mrope_position_ids: [3, batch, seq] with (T, H, W) per-token positions.
    /// image_features:     [num_image_tokens, hidden_size]
    /// deepstack_features: one tensor per layer [num_image_tokens, hidden_size]
    /// image_token_id:     the placeholder token ID used in input_ids for image patches
    pub fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        mrope_position_ids: &Tensor,
        image_features: &Tensor,
        deepstack_features: &[Tensor],
        image_token_id: i64,
    ) -> Tensor {
        let (batch, seq) = input_ids.size2().unwrap();
        let device = input_ids.device();
        let hidden_size = image_features.size()[1];

        let (cos, sin) = self.rotary_emb.forward(mrope_position_ids);

        let flat = input_ids.reshape([-1]);
        let image_mask = flat.eq(image_token_id);

        let mut h = self.embed_tokens.index_select(0, &flat).reshape([-1, hidden_size]);
        h = h.index_put(&[Some(image_mask.shallow_clone())], image_features, false);
        let mut h = h.reshape([batch, seq, hidden_size]);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &cos, &sin);

            if layer_idx < deepstack_features.len() {
                let feat = deepstack_features[layer_idx].to_kind(h.kind()).to_device(device);
                let h_flat = h.reshape([-1, hidden_size]);
                let visual_h = h_flat.index(&[Some(image_mask.shallow_clone())]);
                let updated = visual_h + feat;
                h = h_flat
                    .index_put(&[Some(image_mask.shallow_clone())], &updated, false)
                    .reshape([batch, seq, hidden_size]);
            }
        }

        self.norm.forward(&h)
    }
}
