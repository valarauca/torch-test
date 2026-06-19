use std::path::Path;

use hf_hub::{CacheRepo, api::tokio::ApiRepo};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[repr(u64)]
pub enum ModelIds {
    Qwen3VL = 1,
    Qwen3VLMoe = 2,
    Qwen3VLEmbedding = 3,
    Qwen3VLReranker = 4,
}

/// Infers a `ModelIds` from a local directory by reading its `config.json`.
pub(crate) fn infer_local_path(root: &Path) -> anyhow::Result<Option<ModelIds>> {
    let path = root.join("config.json");
    if !path.try_exists().unwrap_or(false) {
        return Ok(None);
    }
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(classify_config(&raw))
}

/// Infers a `ModelIds` from a locally cached repo by reading its `config.json`.
pub(crate) fn infer_local_repo(repo: &CacheRepo) -> anyhow::Result<Option<ModelIds>> {
    let path = match repo.get("config.json") {
        None => return Ok(None),
        Some(p) => p,
    };
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(classify_config(&raw))
}

/// Infers a `ModelIds` from a remote repo by downloading and reading its `config.json`.
pub(crate) async fn infer_remote_repo(repo: &ApiRepo) -> anyhow::Result<Option<ModelIds>> {
    let path = repo.get("config.json").await?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    Ok(classify_config(&raw))
}

fn classify_config(raw: &serde_json::Value) -> Option<ModelIds> {
    match raw["model_type"].as_str()? {
        "qwen3_vl" => {
            let tc = &raw["text_config"];
            if tc["num_local_experts"].is_number() || tc["num_experts"].is_number() {
                Some(ModelIds::Qwen3VLMoe)
            } else {
                Some(ModelIds::Qwen3VL)
            }
        }
        _ => None,
    }
}
