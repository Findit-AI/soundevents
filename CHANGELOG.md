# Changelog

All notable changes to this workspace will be documented in this file.

## Unreleased

## 0.3.0 - 2026-04-21

### `soundevents`

- Add serde support for `Options` and `ChunkingOptions`.
- Breaking changes:
  - `Options::new` no longer takes a model path; construct options separately from model loading.
  - Chunked classification APIs now take `&ChunkingOptions` instead of `ChunkingOptions`.
  - `ChunkingOptions` is no longer `Copy`; pass it by reference or clone it explicitly where needed.

## 0.2.0 - 2026-04-08

### `soundevents`

- Added `predict_raw_scores_batch_flat` and `predict_raw_scores_batch_into` for lower-allocation batched raw-score access.
- Expanded batched inference coverage with regression tests that verify flat and buffer-reuse paths against sequential inference.
- Removed redundant input validation in `classify_batch` while preserving the existing error behavior for invalid batches.
- Tightened crate metadata and docs.rs configuration so feature-gated APIs, including `Classifier::tiny`, render correctly on published docs.
- Added packaged third-party notices for bundled CED model artifacts.

### `soundevents-dataset`

- Packaged the dual-license texts with the published crate and aligned crate metadata for docs.rs and crates.io discovery.
- Kept the crate on its Rust 1.59 / edition 2021 compatibility track while removing the in-source `deny(warnings)` footgun.
- Added packaged third-party notices for bundled AudioSet ontology and label metadata.

### Workspace

- Included license files in published package contents for both crates.
- Upgraded README examples from ignored snippets to compile-checked doctests across the workspace.

## 0.1.0 - 2026-04-08

### `soundevents`

- Initial public release of the ONNX Runtime wrapper for CED AudioSet classifiers.
- Added file, memory, and bundled-model loading paths plus configurable graph optimization.
- Added ranked top-k helpers, raw-score accessors, and chunked inference with mean/max aggregation.
- Added equal-length batch APIs for clip inference and chunked window batching for higher-throughput services.

### `soundevents-dataset`

- Initial public release of the typed AudioSet dataset companion crate.
- Included both the 527-class rated label set and the full 632-entry ontology as `&'static` generated data.
- Kept the crate `no_std`-friendly, allocation-free at runtime, and compatible with Rust 1.59.

### `xtask`

- Added code generation for the rated label set and ontology modules from upstream AudioSet source data.
