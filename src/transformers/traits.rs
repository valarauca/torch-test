use std::{
    path::{PathBuf},
    future::{Future},
    pin::Pin,
    any::Any,
};

use tokenizers::tokenizer::Tokenizer;
use super::model_ids::{ModelIds};


pub type PinnedFuture<T> = Pin<Box<dyn Future<Output=anyhow::Result<T>> + 'static + Send>>;

/// Something that represents represents the repo
pub trait ModelRepo: 'static + Sync + Send {

    /// What model are we working with
    fn identifier(&self) -> Option<ModelIds>;

    /// Returns the path of this item within a repo.
    ///
    /// Function is `async` as to permit downloading behind the scenes if the model is remote
    fn get_local_path(&self, _name: &str) -> anyhow::Result<Option<PathBuf>>;
}

/// ModelFactory constructs the underly model
pub trait ModelFactory: 'static + Sync {

    /// What model are we working with
    fn identifier(&self) -> Option<ModelIds>;

    /// Load a model.
    fn load(repo: Box<dyn ModelRepo>, _device: tch::Device) -> PinnedFuture<Box<dyn LocalModelBuilder>>;
}

/// Represents a model is "ready to run". 
pub trait LocalModelBuilder: 'static {

    fn identifier(&self) -> Option<ModelIds>;

    fn text_tokenizer(&self) -> Option<Box<dyn TextTokenizer>>;
    fn image_tokenizer(&self) -> Option<Box<dyn ImageTokenizer>>;

    fn is_embedding_model(&self) -> bool;
    /// return a embedding model.
    ///
    /// # confusing return type
    ///
    /// - None is returned when this is not a embedding model
    /// - Some(Err(e)) is returned when this **IS** a embedding model, but an error occured with instruct or while loading the model.
    /// - Some(Ok(x)) is returned when this **IS** a embedding model and it was loaded successfully.
    fn get_embedding_model(&self, query: EmbeddingScheme) -> Option<anyhow::Result<Box<dyn EmbeddingModel>>>;

    fn is_ranking_model(&self) -> bool;
    /// return a ranking model.
    ///
    /// # confusing return type
    ///
    /// - None is returned when this is not a ranking model
    /// - Some(Err(e)) is returned when this **IS** a ranking model, but an error occured with instruct or while loading the model.
    /// - Some(Ok(x)) is returned when this **IS** a ranking model and it was loaded successfully.
    fn get_ranking_model(&self, instruct: EmbeddingScheme) -> Option<anyhow::Result<Box<dyn RankingModel>>>;
}

pub enum EmbeddingScheme {
    NoInput,
    QueryOnly(tch::Tensor),
    QueryAndInstruct(tch::Tensor,tch::Tensor),
}

pub trait TextTokenizer: 'static {
    fn encode(&self, text: &str) -> anyhow::Result<tch::Tensor>;
    fn decode(&self, tokens: tch::Tensor) -> anyhow::Result<String>;
}

pub trait ImageTokenizer: 'static {

    fn encode(&self, img: &image::DynamicImage) -> anyhow::Result<tch::Tensor>;
}


/*
 * Embedding Models
 *
 */

/// Usually a form of sentence transformer
pub trait EmbeddingModel: 'static {
    fn embed(&self, info: tch::Tensor) -> anyhow::Result<tch::Tensor>;

    /// Multimodal embedding: encodes text together with optional image patches.
    ///
    /// pixel_values: packed patches [total_patches, C * t_patch * h_patch * w_patch]
    /// grid_thw: (T, H, W) in patch units for each image
    ///
    /// Falls back to text-only `embed` when pixel_values is None.
    fn embed_multimodal(
        &self,
        input_ids: tch::Tensor,
        pixel_values: Option<tch::Tensor>,
        grid_thw: Option<Vec<(i64, i64, i64)>>,
    ) -> anyhow::Result<tch::Tensor> {
        let _ = (pixel_values, grid_thw);
        self.embed(input_ids)
    }
}

/*
 * Rank Models
 *
 */

/// Model for Ranking inputs
pub trait RankingModel: 'static {
    fn rank(&self, docs: &[tch::Tensor]) -> anyhow::Result<tch::Tensor>;
}
