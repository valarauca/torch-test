use tch::{Kind, Tensor};

use crate::transformers::models::qwen3_vl::dense::Linear;

/// Hyperparameters for the Qwen3-VL vision encoder.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_heads: i64,
    pub in_channels: i64,
    pub patch_size: i64,
    pub spatial_merge_size: i64,
    pub temporal_patch_size: i64,
    pub out_hidden_size: i64,
    pub num_position_embeddings: i64,
    pub deepstack_visual_indexes: Vec<usize>,
}

/// Standard layer normalisation with learnable weight and bias.
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let n = self.weight.size()[0];
        x.layer_norm(&[n], Some(&self.weight), Some(&self.bias), self.eps, true)
    }
}

/// 3-D convolution with a fixed kernel / stride used for patch projection.
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub kernel: [i64; 3],
}

/// Converts raw pixel patches into patch token embeddings via Conv3d.
pub struct VisionPatchEmbed {
    pub proj: Conv3d,
    pub in_channels: i64,
    pub temporal_patch_size: i64,
    pub patch_size: i64,
    pub embed_dim: i64,
}

impl VisionPatchEmbed {
    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let dtype = self.proj.weight.kind();
        let x = hidden_states
            .to_kind(dtype)
            .reshape([-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size]);
        x.conv3d(&self.proj.weight, Some(&self.proj.bias), self.proj.kernel.as_ref(), &[0i64, 0, 0], &[1i64, 1, 1], 1)
            .reshape([-1, self.embed_dim])
    }
}

/// 2-D rotary positional embedding used inside the vision encoder.
/// dim is head_dim // 2; inv_freq has shape [dim // 2].
pub struct VisionRotaryEmbedding {
    pub inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    /// Builds inv_freq = 1 / 10000^(2i/dim) for i in [0, dim/2), matching
    /// PyTorch's `1.0 / (theta ** (arange(0, dim, 2) / dim))` exactly.
    pub fn new(dim: i64, device: tch::Device) -> Self {
        let exponents = Tensor::arange_start_step(0, dim, 2, (Kind::Float, device)).f_div_scalar(dim as f64).unwrap();
        let inv_freq = Tensor::pow_scalar(10000.0, &exponents).reciprocal();
        Self { inv_freq }
    }

    /// Returns flattened frequencies of shape [seq, dim].
    /// position_ids: [seq, 2] with (row, col) coordinates.
    pub fn forward(&self, position_ids: &Tensor) -> Tensor {
        let freqs = position_ids.unsqueeze(-1).to_kind(Kind::Float) * &self.inv_freq;
        freqs.flatten(1, -1)
    }
}

/// SwiGLU-free two-layer feed-forward with GELU-tanh activation.
pub struct VisionMlp {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl VisionMlp {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.fc1.forward(x).gelu("tanh"))
    }
}

fn rotate_half(x: &Tensor) -> Tensor {
    let half = x.size().last().copied().unwrap() / 2;
    Tensor::cat(&[&x.narrow(-1, half, half).neg(), &x.narrow(-1, 0, half)], -1)
}

fn apply_rotary_pos_emb_vision(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> (Tensor, Tensor) {
    let orig_q = q.kind();
    let orig_k = k.kind();
    let qf = q.to_kind(Kind::Float);
    let kf = k.to_kind(Kind::Float);
    let cos = cos.unsqueeze(-2).to_kind(Kind::Float);
    let sin = sin.unsqueeze(-2).to_kind(Kind::Float);
    let q_out = &qf * &cos + rotate_half(&qf) * &sin;
    let k_out = &kf * &cos + rotate_half(&kf) * &sin;
    (q_out.to_kind(orig_q), k_out.to_kind(orig_k))
}

/// Bidirectional multi-head attention for vision tokens.
/// Processes each image independently using cu_seqlens boundaries.
pub struct VisionAttention {
    pub qkv: Linear,
    pub proj: Linear,
    pub num_heads: i64,
    pub head_dim: i64,
}

impl VisionAttention {
    /// cu_seqlens: cumulative sequence offsets [0, len_0, len_0+len_1, ...].
    pub fn forward(&self, hidden: &Tensor, cu_seqlens: &[i64], cos: &Tensor, sin: &Tensor) -> Tensor {
        let seq_len = hidden.size()[0];
        let qkv = self.qkv.forward(hidden).reshape([seq_len, 3, self.num_heads, self.head_dim]).permute([1i64, 0, 2, 3]);
        let q = qkv.select(0, 0);
        let k = qkv.select(0, 1);
        let v = qkv.select(0, 2);

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin);

        let q = q.transpose(0, 1).unsqueeze(0);
        let k = k.transpose(0, 1).unsqueeze(0);
        let v = v.transpose(0, 1).unsqueeze(0);

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut parts: Vec<Tensor> = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));

        for i in 0..cu_seqlens.len().saturating_sub(1) {
            let start = cu_seqlens[i];
            let len = cu_seqlens[i + 1] - start;
            let qi = q.narrow(2, start, len);
            let ki = k.narrow(2, start, len);
            let vi = v.narrow(2, start, len);

            let orig = qi.kind();
            let scores = qi.matmul(&ki.transpose(-2, -1)) * scale;
            let attn = scores.softmax(-1, Kind::Float).to_kind(orig).matmul(&vi);
            parts.push(attn.transpose(1, 2));
        }

        let refs: Vec<&Tensor> = parts.iter().collect();
        let out = Tensor::cat(&refs, 1).reshape([seq_len, -1]);
        self.proj.forward(&out)
    }
}

/// Pre-norm vision transformer block.
pub struct VisionBlock {
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub attn: VisionAttention,
    pub mlp: VisionMlp,
}

impl VisionBlock {
    pub fn forward(&self, hidden: &Tensor, cu_seqlens: &[i64], cos: &Tensor, sin: &Tensor) -> Tensor {
        let h = hidden + self.attn.forward(&self.norm1.forward(hidden), cu_seqlens, cos, sin);
        &h + self.mlp.forward(&self.norm2.forward(&h))
    }
}

/// Spatial patch merger: folds spatial_merge_size² adjacent patches into one token.
/// merged_size = hidden_size * spatial_merge_size².
pub struct PatchMerger {
    pub norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub merged_size: i64,
    pub use_postshuffle_norm: bool,
}

impl PatchMerger {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = if self.use_postshuffle_norm {
            self.norm.forward(&x.reshape([-1, self.merged_size]))
        } else {
            self.norm.forward(x).reshape([-1, self.merged_size])
        };
        self.fc2.forward(&self.fc1.forward(&x).gelu("none"))
    }
}

/// Outputs of the vision encoder.
pub struct VisionOutput {
    /// Pre-merger hidden states: [total_patches, hidden_size].
    pub pre_merger: Tensor,
    /// Merged image tokens: [total_merged_tokens, out_hidden_size].
    pub image_features: Tensor,
    /// One tensor per deepstack layer: [total_merged_tokens, out_hidden_size].
    pub deepstack_features: Vec<Tensor>,
}

/// Full Qwen3-VL vision encoder.
pub struct VisionModel {
    pub patch_embed: VisionPatchEmbed,
    /// Positional embedding table: [num_position_embeddings, hidden_size].
    pub pos_embed: Tensor,
    pub rotary_emb: VisionRotaryEmbedding,
    pub blocks: Vec<VisionBlock>,
    pub merger: PatchMerger,
    pub deepstack_merger_list: Vec<PatchMerger>,
    pub deepstack_visual_indexes: Vec<usize>,
    pub spatial_merge_size: i64,
    pub num_grid_per_side: i64,
    pub hidden_size: i64,
}

impl VisionModel {
    /// Encodes a batch of pre-processed image patches.
    ///
    /// pixel_values: [total_patches, in_channels * temporal_patch_size * patch_size * patch_size]
    /// grid_thw: slice of (T, H, W) in patch units for each image.
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &[(i64, i64, i64)]) -> VisionOutput {
        let device = pixel_values.device();

        let cu_seqlens = cu_seqlens_from_grid_thw(grid_thw);
        let position_ids = vision_position_ids(grid_thw, self.spatial_merge_size, device);
        let (bilinear_idx, bilinear_wt) = bilinear_pos_embed_indices_and_weights(grid_thw, self.num_grid_per_side, self.spatial_merge_size, device);

        let mut hidden = self.patch_embed.forward(pixel_values);

        let pos_emb_flat = self.pos_embed.index_select(0, &bilinear_idx.reshape([-1]));
        let total_patches: i64 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
        let pos_emb_4 = pos_emb_flat.reshape([4, total_patches, self.hidden_size]);
        let pos_embeds = (pos_emb_4 * bilinear_wt.unsqueeze(-1)).sum_dim_intlist([0i64].as_ref(), false, Kind::Float);
        let hidden_kind = hidden.kind();
        hidden = hidden + pos_embeds.to_kind(hidden_kind);

        let rotary = self.rotary_emb.forward(&position_ids);
        let emb = Tensor::cat(&[&rotary, &rotary], -1);
        let (cos, sin) = (emb.cos(), emb.sin());

        let mut deepstack_features: Vec<Tensor> = Vec::new();
        for (layer_num, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &cu_seqlens, &cos, &sin);
            if let Some(pos) = self.deepstack_visual_indexes.iter().position(|&idx| idx == layer_num) {
                deepstack_features.push(self.deepstack_merger_list[pos].forward(&hidden));
            }
        }

        VisionOutput {
            image_features: self.merger.forward(&hidden),
            pre_merger: hidden,
            deepstack_features,
        }
    }
    /// Returns the interpolated positional embeddings for `grid_thw`, shape `[total_patches, hidden_size]`.
    pub(crate) fn bilinear_pos_embeds(&self, grid_thw: &[(i64, i64, i64)], device: tch::Device) -> Tensor {
        let (idx, wt) = bilinear_pos_embed_indices_and_weights(grid_thw, self.num_grid_per_side, self.spatial_merge_size, device);
        let total: i64 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
        let emb = self.pos_embed.index_select(0, &idx.reshape([-1])).reshape([4, total, self.hidden_size]);
        (emb * wt.unsqueeze(-1)).sum_dim_intlist([0i64].as_ref(), false, tch::Kind::Float)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Cumulative sequence lengths from per-image (T, H, W) grids.
/// Returns one boundary per temporal frame, matching the bidirectional
/// attention boundaries expected by the vision encoder.
pub fn cu_seqlens_from_grid_thw(grid_thw: &[(i64, i64, i64)]) -> Vec<i64> {
    let mut out = vec![0i64];
    let mut c = 0i64;
    for &(t, h, w) in grid_thw {
        for _ in 0..t {
            c += h * w;
            out.push(c);
        }
    }
    out
}

/// Position IDs (row, col) for every patch across all images.
/// Patches are emitted in merge-block order: for each (block_row, block_col) group
/// of `spatial_merge_size × spatial_merge_size` patches, inner patches are listed
/// before moving to the next block.  This matches the order produced by the
/// image preprocessor and expected by `PatchMerger`.
/// Returns [total_patches, 2] Int64 tensor on `device`.
pub fn vision_position_ids(grid_thw: &[(i64, i64, i64)], spatial_merge_size: i64, device: tch::Device) -> Tensor {
    let total: i64 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
    let ms = spatial_merge_size;
    let mut buf = Vec::with_capacity((total * 2) as usize);
    for &(t, h, w) in grid_thw {
        for _ in 0..t {
            for br in 0..(h / ms) {
                for bc in 0..(w / ms) {
                    for ir in 0..ms {
                        for ic in 0..ms {
                            buf.push(br * ms + ir);
                            buf.push(bc * ms + ic);
                        }
                    }
                }
            }
        }
    }
    Tensor::from_slice(&buf).reshape([total, 2]).to_device(device)
}

/// Bilinear interpolation indices and weights for the positional embedding lookup.
/// Maps each patch at (h, w) to 4 neighbors in the fixed num_grid_per_side × num_grid_per_side grid.
/// Patches are enumerated in the same merge-block order used by `vision_position_ids` so that
/// the resulting embedding slice aligns with the patch tensor.
///
/// Returns `(indices, weights)` each shaped [4, total_patches].
pub fn bilinear_pos_embed_indices_and_weights(grid_thw: &[(i64, i64, i64)], num_grid_per_side: i64, spatial_merge_size: i64, device: tch::Device) -> (Tensor, Tensor) {
    let n = num_grid_per_side;
    let ms = spatial_merge_size;
    let total: i64 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
    let mut idx_buf = Vec::<i64>::with_capacity((total * 4) as usize);
    let mut wt_buf = Vec::<f32>::with_capacity((total * 4) as usize);

    let cpu = tch::Device::Cpu;
    for &(t, h, w) in grid_thw {
        let hg: Vec<f32> = Vec::try_from(Tensor::linspace(0, n - 1, h, (Kind::Float, cpu))).unwrap();
        let wg: Vec<f32> = Vec::try_from(Tensor::linspace(0, n - 1, w, (Kind::Float, cpu))).unwrap();
        for _ in 0..t {
            for br in 0..(h / ms) {
                for bc in 0..(w / ms) {
                    for ir in 0..ms {
                        for ic in 0..ms {
                            let hi = br * ms + ir;
                            let wi = bc * ms + ic;

                            let h_f = hg[hi as usize];
                            let h0 = h_f as i64;
                            let h1 = (h0 + 1).min(n - 1);
                            let dh = h_f - h0 as f32;

                            let w_f = wg[wi as usize];
                            let w0 = w_f as i64;
                            let w1 = (w0 + 1).min(n - 1);
                            let dw = w_f - w0 as f32;

                            idx_buf.push(h0 * n + w0);
                            idx_buf.push(h0 * n + w1);
                            idx_buf.push(h1 * n + w0);
                            idx_buf.push(h1 * n + w1);

                            wt_buf.push((1.0 - dh) * (1.0 - dw));
                            wt_buf.push((1.0 - dh) * dw);
                            wt_buf.push(dh * (1.0 - dw));
                            wt_buf.push(dh * dw);
                        }
                    }
                }
            }
        }
    }

    let idx = Tensor::from_slice(&idx_buf).reshape([total, 4]).transpose(0, 1).contiguous().to_device(device);
    let wt = Tensor::from_slice(&wt_buf)
        .reshape([total, 4])
        .transpose(0, 1)
        .contiguous()
        .to_kind(Kind::Float)
        .to_device(device);

    (idx, wt)
}
