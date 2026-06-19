use std::path::PathBuf;

use hf_hub::api::tokio::Api;
use pyo3::{
    PyResult, Python,
    ffi::c_str,
    types::{PyAnyMethods, PyDict, PyDictMethods},
};
use tch::{Kind, Tensor, no_grad};
use tokenizers::Tokenizer;

use super::{Qwen3VLEmbeddingFactory, Qwen3VLTokenizedData};
use crate::transformers::repo::init_repo;
use crate::transformers::traits::{EmbeddingScheme, ModelFactory};

const TEST_IMAGE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/transformers/models/qwen3_vl/py_tests/heck overflow.jpeg");

#[tokio::test]
async fn qwen3_dense_forward_matches_python() {
    let path = match std::env::var("QWEN3_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let repo = Api::new().expect("hf hub init failed").model("Qwen/Qwen3-VL-2B-Instruct".to_string());
            let root = repo.get("config.json").await.expect("config.json download failed").parent().unwrap().to_path_buf();
            repo.get("tokenizer.json").await.expect("tokenizer.json download failed");
            repo.get("model.safetensors").await.expect("model.safetensors download failed");
            root
        }
    };
    let tok = Tokenizer::from_file(path.join("tokenizer.json")).expect("tokenizer.json missing");
    let enc = tok.encode("Hello world", false).expect("encode failed");
    let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
    let input_ids = Tensor::from_slice(&ids).reshape([1, ids.len() as i64]);
    let model = super::load_text_model_from_dir(&path).await.expect("failed to load Rust TextModel");
    let rust_hs = no_grad(|| model.forward(&input_ids));
    let rust_vec: Vec<f32> = rust_hs.select(1, -1).squeeze_dim(0).to_kind(Kind::Float).try_into().expect("tensor to vec");
    let py_ids = enc.get_ids().iter().map(|&x| x as u32).collect::<Vec<_>>();
    let py_vec: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let locals = PyDict::new(py);
        locals.set_item("venv_path", concat!(env!("CARGO_MANIFEST_DIR"), "/.venv"))?;
        locals.set_item("model_path", path.to_str().unwrap())?;
        locals.set_item("input_ids", py_ids)?;
        py.run(c_str!(include_str!("py_tests/tests_qwen3_dense_helper.py")), None, Some(&locals))?;
        locals.get_item("result")?.unwrap().extract::<Vec<f32>>()
    })
    .expect("Python inference failed");
    assert_eq!(rust_vec.len(), py_vec.len(), "hidden size mismatch");
    let mismatches: Vec<usize> = rust_vec
        .iter()
        .zip(&py_vec)
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > 1.0e-7)
        .map(|(i, _)| i)
        .collect();
    assert!(mismatches.is_empty(), "{} mismatches; first at [{}]: rust={:.6} py={:.6}", mismatches.len(), mismatches[0], rust_vec[mismatches[0]], py_vec[mismatches[0]]);
}

#[tokio::test]
async fn qwen3_dense_embed_matches_python() {
    let path = match std::env::var("QWEN3_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let api_repo = Api::new().expect("hf hub init failed").model("Qwen/Qwen3-VL-2B-Instruct".to_string());
            let root = api_repo.get("config.json").await.expect("config.json download failed").parent().unwrap().to_path_buf();
            api_repo.get("tokenizer.json").await.expect("tokenizer.json download failed");
            api_repo.get("model.safetensors").await.expect("model.safetensors download failed");
            root
        }
    };
    let repo = init_repo(path.as_path()).await.expect("init_repo failed");
    let factory: &dyn ModelFactory = &Qwen3VLEmbeddingFactory;
    let loader = factory.load(repo).await.expect("factory load failed");
    let builder = loader.initialize(tch::Device::Cpu).expect("initialize failed");
    let tok = builder.text_tokenizer().expect("no text tokenizer");
    let embed = builder.get_embedding_model(EmbeddingScheme::NoInput).unwrap().expect("get embed model failed");
    let data = tok.encode("Hello world").expect("encode failed");
    let py_ids: Vec<i32> = {
        let d = (data.as_ref() as &dyn std::any::Any)
            .downcast_ref::<Qwen3VLTokenizedData>()
            .expect("wrong TokenizedData type");
        Vec::try_from(d.input_ids.flatten(0, -1).to_kind(Kind::Int)).unwrap()
    };
    let rust_vec: Vec<f32> = embed
        .embed(&*data)
        .expect("embed failed")
        .squeeze_dim(0)
        .to_kind(Kind::Float)
        .try_into()
        .expect("tensor to vec");
    let py_vec: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let locals = PyDict::new(py);
        locals.set_item("venv_path", concat!(env!("CARGO_MANIFEST_DIR"), "/.venv"))?;
        locals.set_item("model_path", path.to_str().unwrap())?;
        locals.set_item("input_ids", py_ids)?;
        py.run(c_str!(include_str!("py_tests/tests_qwen3_dense_embed_helper.py")), None, Some(&locals))?;
        locals.get_item("result")?.unwrap().extract::<Vec<f32>>()
    })
    .expect("Python embed inference failed");
    assert_eq!(rust_vec.len(), py_vec.len(), "embedding size mismatch");
    let mismatches: Vec<usize> = rust_vec
        .iter()
        .zip(&py_vec)
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > 1.0e-7)
        .map(|(i, _)| i)
        .collect();
    assert!(mismatches.is_empty(), "{} mismatches; first at [{}]: rust={:.6} py={:.6}", mismatches.len(), mismatches[0], rust_vec[mismatches[0]], py_vec[mismatches[0]]);
}

#[tokio::test]
async fn qwen3_dense_multimodal_embed_matches_python() {
    let path = match std::env::var("QWEN3_MODEL_PATH") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let api_repo = Api::new().expect("hf hub init failed").model("Qwen/Qwen3-VL-2B-Instruct".to_string());
            let root = api_repo.get("config.json").await.expect("config.json download failed").parent().unwrap().to_path_buf();
            api_repo.get("tokenizer.json").await.expect("tokenizer.json download failed");
            api_repo.get("model.safetensors").await.expect("model.safetensors download failed");
            root
        }
    };
    let img = image::open(TEST_IMAGE).expect("failed to open test image");
    let repo = init_repo(path.as_path()).await.expect("init_repo failed");
    let factory: &dyn ModelFactory = &Qwen3VLEmbeddingFactory;
    let loader = factory.load(repo).await.expect("factory load failed");
    let builder = loader.initialize(tch::Device::Cpu).expect("initialize failed");
    let img_tok = builder.image_tokenizer().expect("no image tokenizer");
    let embed = builder.get_embedding_model(EmbeddingScheme::NoInput).unwrap().expect("get embed model failed");
    let data = img_tok.encode(&img).expect("image encode failed");
    let d = (data.as_ref() as &dyn std::any::Any)
        .downcast_ref::<Qwen3VLTokenizedData>()
        .expect("wrong TokenizedData type");
    let py_ids: Vec<i32> = Vec::try_from(d.input_ids.flatten(0, -1).to_kind(Kind::Int)).unwrap();
    let py_pv: Vec<f32> = Vec::try_from(d.pixel_values.as_ref().unwrap().to_kind(Kind::Float).flatten(0, -1)).unwrap();
    let (gt, gh, gw) = d.grid_thw.as_ref().unwrap()[0];
    let rust_vec: Vec<f32> = embed
        .embed(&*data)
        .expect("embed failed")
        .squeeze_dim(0)
        .to_kind(Kind::Float)
        .try_into()
        .expect("tensor to vec");
    let py_vec: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let locals = PyDict::new(py);
        locals.set_item("venv_path", concat!(env!("CARGO_MANIFEST_DIR"), "/.venv"))?;
        locals.set_item("model_path", path.to_str().unwrap())?;
        locals.set_item("input_ids", py_ids)?;
        locals.set_item("pixel_values_flat", py_pv)?;
        locals.set_item("grid_thw", (gt, gh, gw))?;
        py.run(c_str!(include_str!("py_tests/tests_qwen3_dense_multimodal_embed_helper.py")), None, Some(&locals))?;
        locals.get_item("result")?.unwrap().extract::<Vec<f32>>()
    })
    .expect("Python multimodal embed inference failed");
    assert_eq!(rust_vec.len(), py_vec.len(), "embedding size mismatch");
    let mismatches: Vec<usize> = rust_vec
        .iter()
        .zip(&py_vec)
        .enumerate()
        .filter(|(_, (a, b))| (*a - *b).abs() > 1.0e-7)
        .map(|(i, _)| i)
        .collect();
    assert!(mismatches.is_empty(), "{} mismatches; first at [{}]: rust={:.6} py={:.6}", mismatches.len(), mismatches[0], rust_vec[mismatches[0]], py_vec[mismatches[0]]);
}
