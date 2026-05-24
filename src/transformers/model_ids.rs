

#[derive(Clone,Copy,PartialEq,Eq,PartialOrd,Ord,Debug,Hash)]
#[repr(u64)]
pub enum ModelIds {
    Qwen3VL = 1,
    Qwen3VLMoe = 2,
    Qwen3VLEmbedding = 3,
    Qwen3VLReranker = 4,
}
