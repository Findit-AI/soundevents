# Third-Party Notices for `soundevents`

This crate redistributes third-party model artifacts in addition to the
project's own Rust source code.

## CED model artifacts

The published crate bundles `models/tiny.onnx`, an ONNX export of the CED-Tiny
audio classification model.

- Upstream model card: <https://huggingface.co/mispeech/ced-tiny>
- Upstream repository referenced by the model card:
  <https://github.com/RicherMans/CED>
- Paper referenced by the model card:
  <https://arxiv.org/abs/2308.11957>
- License reported by the upstream model card at the time this crate was
  packaged: Apache-2.0

The repository may also contain additional CED ONNX variants for development
or benchmarking. See the upstream model cards for the full provenance and
license terms of those artifacts.
