<div align="center">
<h1>soundevents</h1>
</div>
<div align="center">

Production-oriented Rust inference for [CED](https://arxiv.org/abs/2308.11957) AudioSet sound-event classifiers — load an ONNX model, feed it 16 kHz mono audio, get back ranked [`RatedSoundEvent`](./soundevents-dataset) predictions with names, ids, and confidences. Long clips are handled via configurable chunking.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/soundevents-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fsoundevents" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/soundevents/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/soundevents?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## Highlights

- **Drop-in CED inference** — load any [CED](https://arxiv.org/abs/2308.11957) AudioSet ONNX model (or use the bundled `tiny` variant) and run it directly on `&[f32]` PCM samples. No Python, no preprocessing pipeline.
- **Typed labels, not bare integers** — every prediction comes back as an [`EventPrediction`] carrying a `&'static RatedSoundEvent` from [`soundevents-dataset`](./soundevents-dataset), so you get the canonical AudioSet name, the `/m/...` id, the model class index, and the confidence in one struct.
- **Compile-time class-count guarantee** — the `NUM_CLASSES = 527` constant comes from the rated dataset at codegen time. If a model returns the wrong number of classes you get a typed [`ClassifierError::UnexpectedClassCount`] instead of a silent mismatch.
- **Long-clip chunking built in** — `classify_chunked` / `classify_all_chunked` window the input at a configurable hop, run inference on each chunk, and aggregate the per-chunk confidences with either `Mean` or `Max`. Defaults match CED's 10 s training window (160 000 samples at 16 kHz).
- **Top-k via a tiny min-heap** — `classify(samples, k)` does not allocate a full 527-element scores vector to find the top results.
- **Bring-your-own model or bundle one** — load from a path, from in-memory bytes, or enable the `bundled-tiny` feature to embed `models/tiny.onnx` directly into your binary.

## Quick start

```toml
[dependencies]
soundevents = "0.1"
```

```rust,no_run
use soundevents::{Classifier, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::from_file("soundevents/models/tiny.onnx")?;

    // Bring your own decoder/resampler — soundevents expects mono f32
    // samples at 16 kHz, in [-1.0, 1.0].
    let samples: Vec<f32> = load_mono_16k_audio("clip.wav")?;

    // Top-5 predictions for a clip up to ~10 s long.
    for prediction in classifier.classify(&samples, 5)? {
        println!(
            "{:>5.1}%  {:>3}  {}  ({})",
            prediction.confidence() * 100.0,
            prediction.index(),
            prediction.name(),
            prediction.id(),
        );
    }
    Ok(())
}
# fn load_mono_16k_audio(_: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> { Ok(vec![]) }
```

### Long clips: chunked inference

`Classifier::classify_chunked` slides a window over the input and aggregates each chunk's per-class confidences. The defaults (10 s window, 10 s hop, mean aggregation) match CED's training setup; tune them for overlap or peak-pooling.

```rust,no_run
use soundevents::{ChunkAggregation, ChunkingOptions, Classifier};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::from_file("soundevents/models/tiny.onnx")?;
    let samples: Vec<f32> = load_long_clip()?;

    let opts = ChunkingOptions::default()
        // 5 s overlap (50%) between adjacent windows
        .with_hop_samples(80_000)
        // Keep the loudest detection in any window instead of averaging
        .with_aggregation(ChunkAggregation::Max);

    let top3 = classifier.classify_chunked(&samples, 3, opts)?;
    for prediction in top3 {
        println!("{}: {:.2}", prediction.name(), prediction.confidence());
    }
    Ok(())
}
# fn load_long_clip() -> Result<Vec<f32>, Box<dyn std::error::Error>> { Ok(vec![]) }
```

## Models

The four CED variants are sourced from the [`mispeech`](https://huggingface.co/mispeech) Hugging Face organisation, exported to ONNX, and **checked into this repo** under [`soundevents/models/`](./soundevents/models). You should not normally need to download anything — `git clone` gives you a working classifier out of the box.

| Variant | File | Size | Hugging Face source |
| --- | --- | --- | --- |
| `tiny` | `soundevents/models/tiny.onnx` | 6.4 MB | [`mispeech/ced-tiny`](https://huggingface.co/mispeech/ced-tiny) |
| `mini` | `soundevents/models/mini.onnx` | 10 MB | [`mispeech/ced-mini`](https://huggingface.co/mispeech/ced-mini) |
| `small` | `soundevents/models/small.onnx` | 22 MB | [`mispeech/ced-small`](https://huggingface.co/mispeech/ced-small) |
| `base` | `soundevents/models/base.onnx` | 97 MB | [`mispeech/ced-base`](https://huggingface.co/mispeech/ced-base) |

All four expose the same input/output contract: mono `f32` PCM at 16 kHz in, 527-class scores out (`SAMPLE_RATE_HZ` / `NUM_CLASSES`). They differ only in parameter count and accuracy/latency trade-off, so you can swap variants without touching application code.

> **Note** — the four ONNX files together are ~135 MB. If you fork this repo and want to keep the working tree slim, consider tracking `soundevents/models/*.onnx` with [git LFS](https://git-lfs.com/).

### Refreshing models from upstream

If upstream releases new weights, or you cloned without the model files, refetch them with:

```sh
# Requires huggingface_hub:  pip install --user huggingface_hub
./scripts/download_models.sh

# Or just one variant
./scripts/download_models.sh tiny
```

The script downloads the `*.onnx` artifact from each `mispeech/ced-*` Hugging Face repo and writes it as `soundevents/models/<variant>.onnx`.

### Bundled tiny model

Enable the `bundled-tiny` feature to embed `models/tiny.onnx` into your binary — useful for CLI tools and self-contained services where you don't want to ship a separate model file.

```toml
soundevents = { version = "0.1", features = ["bundled-tiny"] }
```

```rust,ignore
use soundevents::{Classifier, Options};

let mut classifier = Classifier::tiny(Options::default())?;
```

## Features

| Feature | Default | What you get |
| --- | :-: | --- |
| `bundled-tiny` | | Embeds `models/tiny.onnx` into the crate so `Classifier::tiny()` works without an external file. |

The full input/output contract:

| Constant | Value | Meaning |
| --- | --- | --- |
| `SAMPLE_RATE_HZ` | `16_000` | Required input sample rate (mono `f32`). |
| `DEFAULT_CHUNK_SAMPLES` | `160_000` | Default 10 s window/hop for chunked inference. |
| `NUM_CLASSES` | `527` | Number of CED output classes — derived at compile time from `RatedSoundEvent::events().len()`. |

## Development

Regenerate the dataset from upstream sources:

```sh
cargo xtask codegen
```

Run the test suite:

```sh
cargo test
```

[`EventPrediction`]: https://docs.rs/soundevents/latest/soundevents/struct.EventPrediction.html
[`ClassifierError::UnexpectedClassCount`]: https://docs.rs/soundevents/latest/soundevents/enum.ClassifierError.html

#### License

`soundevents` is under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 FinDIT studio authors.

[Github-url]: https://github.com/Findit-AI/soundevents
[CI-url]: https://github.com/Findit-AI/soundevents/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/Findit-AI/soundevents/
