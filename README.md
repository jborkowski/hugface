# HuggingFace Rust Sandbox

This repository serves as a sandbox for experimenting with HuggingFace's Rust library, particularly focusing on Language Model capabilities. The project is based on the Shuttle.rs's blog post ["Using HuggingFace Rust"](https://www.shuttle.rs/blog/2024/05/01/using-huggingface-rust).

## How to Use
1. Clone the repository to your local machine.
2. Add `.env` file with `HF_TOKEN` from [HuggingFace](https://huggingface.co) 
3. Run `cargo run --release`
4. Make HTTP calls with prompts: `http POST localhost:8000/prompt prompt='How are you?</s>'`

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

---

*Note: This repository is for educational and experimental purposes only. It is not intended for production use.*

