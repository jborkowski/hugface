use candle_core::{DType, Device, Error, Result, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::mistral::Model as Mistral};
use tokenizers::Tokenizer;

use crate::model::AppState;

pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        TokenOutputStream {
            tokenizer,
            tokens: vec![],
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => {
                candle_core::bail!("cannot decode: {err}")
            }
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };

        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphabetic() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

pub struct TextGeneration {
    model: Mistral,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    /// Constructs a new instance of the model with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `model` - The Mistral model to be used.
    /// * `tokenizer` - The Tokenizer for preprocessing input data.
    /// * `seed` - The seed value for random number generation.
    /// * `temp` - Optional parameter to adjust the temperature for controlling the accuracy of the model's output. Lower values result in less hallucinated output.
    /// * `top_p` - Optional parameter to specify a probability threshold for token selection, ensuring diverse responses while maintaining context relevance.
    /// * `_top_k` - Optional parameter to specify the number of top tokens to sample from, making the model more predictable and consistent.
    /// * `repeat_penalty` - The penalty factor for redundant or repetitive output.
    /// * `repeat_last_n` - The size of the context window for determining repetitive output. Tokens can vary in size from 1 to 4 words.
    /// * `device` - The target device for model computation.
    ///
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Mistral,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        _top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(mut self, prompt: String, sample_len: usize) -> String {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => panic!("cannot find the </s> token"),
        };

        let mut string = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.forward(&input, start_pos).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                println!("Found a token!");
                string.push_str(&t);
            }
        }

        string
    }
}

impl From<AppState> for TextGeneration {
    fn from(s: AppState) -> Self {
        Self::new(s.model, s.tokenizer, 598797954, Some(0.), None, None, 1.1, 64, &s.device)
    }
}
