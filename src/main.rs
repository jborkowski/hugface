mod llm;
mod model;

use anyhow::{Context, Result};

use axum::{extract::State, response::IntoResponse, routing::{get, post}, Json, Router};
use dotenv::dotenv;
use llm::TextGeneration;
use model::AppState;
use serde::Deserialize;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let token = std::env::var("HF_TOKEN").context(
        "Failed to retrieve HuggingFace token. The environment variable 'HF_TOKEN' is not found."
            .to_string(),
    )?;

    let state = model::initialise_model(token)?;

    let router = Router::new()
        .route("/", get(hello_world))
        .route("/prompt", post(run_pipeline))
        .with_state(state);

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:8000")
        .await
        .unwrap();
    axum::serve(tcp_listener, router).await.unwrap();

    Ok(())
}

async fn hello_world() -> &'static str {
    "Hello, world!"
}

#[derive(Deserialize)]
pub struct Prompt {
    prompt: String,
}

async fn run_pipeline(
    State(state): State<AppState>,
    Json(Prompt { prompt }): Json<Prompt>,
) -> impl IntoResponse {
    let textgen = TextGeneration::from(state);
    textgen.run(prompt, 2000)
}
