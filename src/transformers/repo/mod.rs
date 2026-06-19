use std::path::{Path, PathBuf};

use hf_hub::{Repo as HFRepo, api::tokio::Api};
use local::LocalRepo;
use remote::RemoteRepo;

use crate::transformers::traits::ModelRepo;

pub mod local;
pub mod remote;

pub enum Repo {
    Local(PathBuf),
    // assumed to be remote.
    Repo(HFRepo),
}
impl From<&Path> for Repo {
    fn from(p: &Path) -> Repo {
        Repo::Local(p.to_owned())
    }
}
impl From<&HFRepo> for Repo {
    fn from(r: &HFRepo) -> Repo {
        Repo::Repo(r.clone())
    }
}

/// Initialize a repo
pub async fn init_repo<T>(arg: T) -> anyhow::Result<Box<dyn ModelRepo>>
where
    Repo: From<T>,
{
    match Repo::from(arg) {
        Repo::Local(path) => Ok(Box::new(LocalRepo::new(path)?)),
        Repo::Repo(hf) => Ok(Box::new(RemoteRepo::new(Api::new()?.repo(hf)).await?)),
    }
}
