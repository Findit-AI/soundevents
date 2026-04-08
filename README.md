<div align="center">
<h1>soundevents</h1>
</div>
<div align="center">

Production-oriented Rust inference for [CED](https://arxiv.org/abs/2308.11957) AudioSet sound-event classifiers — load an ONNX model, feed it 16 kHz mono audio, get back ranked [`RatedSoundEvent`](./soundevents-dataset) predictions with names, ids, and confidences. Long clips are handled via configurable chunking.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/soundevents-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fsoundevents" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/soundevents/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/soundevents?style=for-the-badge&token=6R3QFWRWHL&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-soundevents-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/soundevents?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPC9zdmc+DQo=" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/soundevents?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge" height="22">

</div>

## Highlights

- **Drop-in CED inference** — load any [CED](https://arxiv.org/abs/2308.11957) AudioSet ONNX model (or use the bundled `tiny` variant) and run it directly on `&[f32]` PCM samples. No Python, no preprocessing pipeline.
- **Typed labels, not bare integers** — every prediction comes back as an [`EventPrediction`] carrying a `&'static RatedSoundEvent` from [`soundevents-dataset`](./soundevents-dataset), so you get the canonical AudioSet name, the `/m/...` id, the model class index, and the confidence in one struct.
- **Compile-time class-count guarantee** — the `NUM_CLASSES = 527` constant comes from the rated dataset at codegen time. If a model returns the wrong number of classes you get a typed [`ClassifierError::UnexpectedClassCount`] instead of a silent mismatch.
- **Long-clip chunking built in** — `classify_chunked` / `classify_all_chunked` window the input at a configurable hop, run inference on each chunk, and aggregate the per-chunk confidences with either `Mean` or `Max`. Defaults match CED's 10 s training window (160 000 samples at 16 kHz), and fixed-size chunk batches can now be packed into one model call.
- **Top-k via a tiny min-heap** — `classify(samples, k)` does not allocate a full 527-element scores vector to find the top results.
- **Batch-ready low-level API** — `predict_raw_scores_batch`, `predict_raw_scores_batch_flat`, `predict_raw_scores_batch_into`, `classify_all_batch`, and `classify_batch` accept equal-length clip batches for service-layer batching.
- **Bring-your-own model or bundle one** — load from a path, from in-memory bytes, or enable the `bundled-tiny` feature to embed `models/tiny.onnx` directly into your binary.

## Quick start

```toml
[dependencies]
soundevents = "0.2"
```

```rust,no_run
use soundevents::{Classifier, Options};

fn load_mono_16k_audio(_: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(vec![0.0; 16_000])
}

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
```

### Long clips: chunked inference

`Classifier::classify_chunked` slides a window over the input and aggregates each chunk's per-class confidences. The defaults (10 s window, 10 s hop, mean aggregation) match CED's training setup; tune them for overlap or peak-pooling.

```rust,no_run
use soundevents::{ChunkAggregation, ChunkingOptions, Classifier};

fn load_long_clip() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(vec![0.0; 320_000])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut classifier = Classifier::from_file("soundevents/models/tiny.onnx")?;
    let samples: Vec<f32> = load_long_clip()?;

    let opts = ChunkingOptions::default()
        // 5 s overlap (50%) between adjacent windows
        .with_hop_samples(80_000)
        // Batch up to 4 equal-length windows per session.run()
        .with_batch_size(4)
        // Keep the loudest detection in any window instead of averaging
        .with_aggregation(ChunkAggregation::Max);

    let top3 = classifier.classify_chunked(&samples, 3, opts)?;
    for prediction in top3 {
        println!("{}: {:.2}", prediction.name(), prediction.confidence());
    }
    Ok(())
}
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

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream model
sources and attribution details.

### Bundled tiny model

Enable the `bundled-tiny` feature to embed `models/tiny.onnx` into your binary — useful for CLI tools and self-contained services where you don't want to ship a separate model file.

```toml
soundevents = { version = "0.2", features = ["bundled-tiny"] }
```

```rust
# #[cfg(feature = "bundled-tiny")]
use soundevents::{Classifier, Options};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
# #[cfg(feature = "bundled-tiny")]
# {
let mut classifier = Classifier::tiny(Options::default())?;
# let _ = &mut classifier;
# }
# Ok(())
# }
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

For low-level batching, every clip in `predict_raw_scores_batch*` / `classify_*_batch` must be non-empty and have the same sample count. `predict_raw_scores_batch_flat` returns one row-major `Vec<f32>`, and `predict_raw_scores_batch_into` lets callers reuse their own output buffer to avoid per-call result allocations. `classify_chunked` uses the same equal-length restriction internally when `ChunkingOptions::batch_size() > 1`, which is naturally satisfied for fixed-size windows and automatically falls back to smaller batches for the final short tail chunk.

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
Bundled third-party model attributions and source licenses are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Copyright (c) 2026 FinDIT studio authors.

[Github-url]: https://github.com/Findit-AI/soundevents
[CI-url]: https://github.com/Findit-AI/soundevents/actions/workflows/ci.yml
[codecov-url]: https://app.codecov.io/gh/Findit-AI/soundevents/
[doc-url]: https://docs.rs/soundevents
[crates-url]: https://crates.io/crates/soundevents
