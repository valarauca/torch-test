use std::{
    future::{Future, ready},
    path::PathBuf,
    pin::Pin,
};

use crate::transformers::{
    model_ids::{ModelIds, infer_local_path},
    traits::ModelRepo,
};

/// A `ModelRepo` backed by a local directory.
pub struct LocalRepo {
    root: PathBuf,
    id: Option<ModelIds>,
}

impl LocalRepo {
    pub(crate) fn new(root: PathBuf) -> anyhow::Result<Self> {
        let id = infer_local_path(&root)?;
        Ok(Self { root, id })
    }
}

impl ModelRepo for LocalRepo {
    fn identifier(&self) -> Option<ModelIds> {
        self.id
    }

    fn get_local_path<'a>(&'a self, name: &'a str) -> Pin<Box<dyn Future<Output = anyhow::Result<Option<PathBuf>>> + 'a>> {
        let path = self.root.join(name);
        let out = if path.try_exists().unwrap_or(false) { Ok(Some(path)) } else { Ok(None) };
        Box::pin(ready(out))
    }
}
