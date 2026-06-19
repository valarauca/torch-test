use std::{any::Any, future::Future, path::PathBuf, pin::Pin};

use super::model_ids::ModelIds;

pub type PinnedFuture<T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + 'static + Send>>;

/// Something that represents represents the repo
pub trait ModelRepo: 'static + Sync + Send {
    /// What model are we working with
    fn identifier(&self) -> Option<ModelIds>;

    /// Returns the path of this item within a repo.
    ///
    /// Function is `async` as to permit downloading behind the scenes if the model is remote
    fn get_local_path<'a>(&'a self, name: &'a str) -> Pin<Box<dyn Future<Output = anyhow::Result<Option<PathBuf>>> + 'a>>;
}

/// ModelFactory constructs the underly model
pub trait ModelFactory: 'static + Sync + Send {
    /// What model are we working with
    fn identifier(&self) -> Option<ModelIds>;

    /// Load a model.
    fn load<'a>(&'a self, repo: Box<dyn ModelRepo>) -> Pin<Box<dyn Future<Output = anyhow::Result<Box<dyn ModelLoader>>> + 'a>>;
}

/// Handles loading a model's tensors into memory
pub trait ModelLoader: 'static + Send {
    /// Returns the model variant this loader was constructed for.
    fn identifier(&self) -> Option<ModelIds>;
    /// Allocates tensors onto `device` and returns a ready to run model.
    fn initialize(&self, _device: tch::Device) -> anyhow::Result<Box<dyn LocalModelBuilder>>;
}

/// Represents a model is "ready to run".
/// When this type is dropped all tensors should be unloaded.
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
    QueryAndInstruct(tch::Tensor, tch::Tensor),
}

pub trait TextTokenizer: 'static {
    fn encode(&self, text: &str) -> anyhow::Result<Box<dyn TokenizedData>>;

    /// TODO: eventually this'll require a classifier.
    fn decode(&self, tokens: tch::Tensor) -> anyhow::Result<String>;
}

pub trait ImageTokenizer: 'static {
    fn encode(&self, img: &image::DynamicImage) -> anyhow::Result<Box<dyn TokenizedData>>;
}

/// General wrapper for model tokenized data
pub trait TokenizedData: 'static + Any {}

/*
 * Embedding Models
 *
 */

/// Usually a form of sentence transformer
pub trait EmbeddingModel: 'static {
    /// It is required TokenizedData comes from Tokenizes produced by 'this' model
    fn embed(&self, info: &dyn TokenizedData) -> anyhow::Result<tch::Tensor>;
}

/*
 * Rank Models
 *
 */

/// Model for Ranking inputs
pub trait RankingModel: 'static {
    /// Scores each tokenized query-document pair.  Each `TokenizedData` may carry
    /// optional vision tensors, so ranking covers text-only and multimodal inputs.
    fn rank(&self, docs: &[&dyn TokenizedData]) -> anyhow::Result<tch::Tensor>;
}
