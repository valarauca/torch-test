use std::{
    collections::HashSet,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
};

use hf_hub::api::tokio::ApiRepo;

use crate::transformers::{
    model_ids::{ModelIds, infer_remote_repo},
    traits::ModelRepo,
};

/// A `ModelRepo` backed by the HuggingFace Hub async API.
pub struct RemoteRepo {
    id: Option<ModelIds>,
    contents: HashSet<PathBuf>,
    repo: ApiRepo,
}

impl RemoteRepo {
    /// Fetches repo metadata upfront so subsequent `get_local_path` calls are non-blocking.
    pub(crate) async fn new(repo: ApiRepo) -> anyhow::Result<Self> {
        let id = infer_remote_repo(&repo).await?;
        let data = repo.info().await?;
        let contents = data.siblings.into_iter().map(|s| PathBuf::from(s.rfilename)).collect();
        Ok(Self { id, repo, contents })
    }
}

impl ModelRepo for RemoteRepo {
    fn identifier(&self) -> Option<ModelIds> {
        self.id
    }

    fn get_local_path<'a>(&'a self, name: &'a str) -> Pin<Box<dyn Future<Output = anyhow::Result<Option<PathBuf>>> + 'a>> {
        Box::pin(async move {
            if !self.contents.contains(Path::new(name)) {
                return Ok(None);
            }
            Ok(Some(self.repo.get(name).await?))
        })
    }
}
