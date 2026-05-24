use tch::{Kind, Tensor};

/// Hyperparameters for the text transformer.
#[derive(Clone, Debug)]
pub struct TextConfig {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 22016,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 500_000.0,
        }
    }
}

/// RMS layer normalisation: weight * x * rsqrt(mean(x²) + ε).
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    /// Upcasts to f32 for the variance computation, then casts the result back.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let orig = x.kind();
        let xf = x.to_kind(Kind::Float);
        let var = xf.pow_tensor_scalar(2.0).mean_dim([-1i64].as_ref(), true, Kind::Float);
        let normed = xf * (var + self.eps).rsqrt();
        self.weight.to_kind(orig) * normed.to_kind(orig)
    }
}

/// Weight-only linear projection (bias optional).
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    /// y = x Wᵀ (+ bias)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.linear(&self.weight, self.bias.as_ref())
    }
}

/// RoPE inverse-frequency table, computed once at load time.
pub struct RotaryEmbedding {
    pub inv_freq: Tensor,
}

impl RotaryEmbedding {
    /// inv_freq[i] = 1 / θ^(2i / head_dim)
    pub fn new(head_dim: i64, theta: f64, device: tch::Device) -> Self {
        // i = 0, 1, ..., head_dim/2 - 1  →  exponent = 2i/head_dim
        let exponents = Tensor::arange(head_dim / 2, (Kind::Float, device))
            .f_mul_scalar(2.0 / head_dim as f64).unwrap();
        let inv_freq = exponents.f_mul_scalar(theta.ln()).unwrap().exp().reciprocal();
        Self { inv_freq }
    }

    /// Returns (cos, sin) both shaped [batch, seq, head_dim].
    pub fn forward(&self, position_ids: &Tensor) -> (Tensor, Tensor) {
        let (batch, _seq) = position_ids.size2().unwrap();
        // inv_freq: [d/2]  →  [batch, d/2, 1]
        let inv = self.inv_freq.unsqueeze(0).unsqueeze(-1).expand([batch, -1, 1], false);
        // position_ids: [batch, seq]  →  [batch, 1, seq]
        let pos = position_ids.unsqueeze(1).to_kind(Kind::Float);
        // freqs: [batch, d/2, seq]  →  transpose  →  [batch, seq, d/2]
        let freqs = inv.matmul(&pos).transpose(1, 2);
        let emb = Tensor::cat(&[&freqs, &freqs], -1);
        (emb.cos(), emb.sin())
    }
}

fn rotate_half(x: &Tensor) -> Tensor {
    let half = x.size().last().copied().unwrap() / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&x2.neg(), &x1], -1)
}

/// Applies RoPE to query and key.  cos/sin: [batch, seq, head_dim].
pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> (Tensor, Tensor) {
    let cos = cos.unsqueeze(1);
    let sin = sin.unsqueeze(1);
    (q * &cos + rotate_half(q) * &sin, k * &cos + rotate_half(k) * &sin)
}

/// Expands key/value heads to match query head count (grouped query attention).
pub fn repeat_kv(x: &Tensor, n_rep: i64) -> Tensor {
    if n_rep == 1 {
        return x.shallow_clone();
    }
    let (batch, n_kv, seq, head_dim) = x.size4().unwrap();
    x.unsqueeze(2)
        .expand([batch, n_kv, n_rep, seq, head_dim], false)
        .reshape([batch, n_kv * n_rep, seq, head_dim])
}

/// SwiGLU feed-forward: down(silu(gate(x)) * up(x)).
pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.down_proj.forward(&(self.gate_proj.forward(x).silu() * self.up_proj.forward(x)))
    }
}

/// Multi-head attention with GQA and per-head Q/K RMS normalisation.
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub num_heads: i64,
    pub num_kv_heads: i64,
    pub head_dim: i64,
}

impl Attention {
    /// Scaled dot-product attention with causal mask.
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (batch, seq, _) = x.size3().unwrap();
        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = (self.head_dim as f64).powf(-0.5);
        let device = x.device();

        let q = self.q_norm
            .forward(&self.q_proj.forward(x).reshape([batch, seq, self.num_heads, self.head_dim]))
            .transpose(1, 2);
        let k = self.k_norm
            .forward(&self.k_proj.forward(x).reshape([batch, seq, self.num_kv_heads, self.head_dim]))
            .transpose(1, 2);
        let v = self.v_proj.forward(x)
            .reshape([batch, seq, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin);
        let k = repeat_kv(&k, n_rep);
        let v = repeat_kv(&v, n_rep);

        let causal_mask = Tensor::ones([seq, seq], (Kind::Bool, device))
            .tril(0)
            .logical_not()
            .unsqueeze(0)
            .unsqueeze(0);

        let orig_kind = q.kind();
        let logits = q.matmul(&k.transpose(-2, -1)) * scale;
        let logits = logits.masked_fill(&causal_mask, f64::NEG_INFINITY);
        let attn = logits
            .softmax(-1, Kind::Float)
            .to_kind(orig_kind)
            .matmul(&v);

        let out = attn.transpose(1, 2).contiguous().reshape([batch, seq, -1]);
        self.o_proj.forward(&out)
    }
}

/// Pre-norm decoder layer with residual connections.
pub struct DecoderLayer {
    pub attn: Attention,
    pub mlp: Mlp,
    pub input_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
}

impl DecoderLayer {
    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let attn_out = self.attn.forward(&self.input_norm.forward(x), cos, sin);
        let h = x + &attn_out;
        let mlp_out = self.mlp.forward(&self.post_attn_norm.forward(&h));
        &h + &mlp_out
    }
}

/// Full Qwen3-VL text transformer (text-only path, no vision tokens).
pub struct TextModel {
    pub embed_tokens: Tensor,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub rotary_emb: RotaryEmbedding,
}

impl TextModel {
    /// Returns final hidden states shaped [batch, seq, hidden_size].
    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let (batch, seq) = input_ids.size2().unwrap();
        let device = input_ids.device();

        let position_ids = Tensor::arange(seq, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([batch, -1], false);
        let (cos, sin) = self.rotary_emb.forward(&position_ids);

        let flat = input_ids.reshape([-1]);
        let mut h = self.embed_tokens.index_select(0, &flat).reshape([batch, seq, -1]);

        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin);
        }

        self.norm.forward(&h)
    }
}
