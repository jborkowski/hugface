# HuggingFace Rust Sandbox

This repository serves as a sandbox for experimenting with HuggingFace's Rust library, particularly focusing on Language Model capabilities. The project is based on the Shuttle.rs's blog post ["Using HuggingFace Rust"](https://www.shuttle.rs/blog/2024/05/01/using-huggingface-rust).

## How to Use
1. Clone the repository to your local machine.
2. Add `.env` file with `HF_TOKEN` from [HuggingFace](https://huggingface.co) 
3. Run `cargo run --release`
4. Make HTTP calls with prompts: `http POST localhost:8000/prompt prompt='How are you?</s>'`

### Example call

```sh
λ http POST localhost:8000/prompt prompt='How are you?  </s>'

м

I’m fine, thanks.  I hope you are too.

I’ve been thinking about the word “fine” lately.  It’s a word that we use all the time to describe how we feel.  But what does it really mean?  Is it just a polite way of saying “I’m not great, but I’m not terrible either”?  Or is there more to it than that?

When you ask someone how they are doing, and they say “fine”, what do you think they mean?  Do you take them at their word, or do you wonder if there’s something else going on beneath the surface?

I think that when we use the word “fine” to describe our feelings, we often don’t really mean it.  We say it because we don’t want to burden others with our problems, or because we don’t want to seem like we’re complaining.  But in reality, most of us are not actually fine all the time.

So what should we do instead?  I think that we need to be more honest with ourselves and with others about how we really feel.  If you’re having a bad day, it’s okay to say so.  And if someone asks you how you are doing, don’t just say “fine” – tell them the truth.

Of course, there will be times when we need to put on a brave face and pretend that everything is fine even when it isn’t.  But those should be the exception, not the rule.  We all have bad days sometimes, but we shouldn’t let them define us.

So next time someone asks you how you are doing, don’t just say “fine”.  Tell them how you really feel – even if it’s not always positive.  And remember that it’s okay to not be fine all the time.

I hope this helps!   Thanks for reading
```

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

---

*Note: This repository is for educational and experimental purposes only. It is not intended for production use.*

