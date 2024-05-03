use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use derive_new::new;
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use serde::de::Deserializer;
use serde::Deserialize;
use tokenizers::Tokenizer;
use std::collections::HashSet;

use candle_transformers::models::mistral::{Config, Model as Mistral};


#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

#[derive(Clone, new)]
pub struct AppState {
    pub model: Mistral,
    pub device: Device,
    pub tokenizer: Tokenizer,
}

fn deserialize_weight_map<'de, D>(deserialize: D) -> Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserialize)?;

    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight map",
        )),
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();
    Ok(pathbufs)
}

fn get_repo(token: String) -> Result<ApiRepo> {
    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();

    Ok(api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "26bca36bde8333b5d7f72e9ed20ccda6a618af24".to_string(),
    )))
}

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

pub fn initialise_model(token: String) -> Result<AppState> {
    let repo = get_repo(token)?;
    let tokenizer = get_tokenizer(&repo)?;
    let device = Device::Cpu;
    let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;


    let config = Config::config_7b_v0_1(false);

    let model = {
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        Mistral::new(&config, vb)?
    };

    Ok(AppState::new(model, device, tokenizer))     
}


