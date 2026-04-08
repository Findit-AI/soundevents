#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use ort::{
  session::{Session, builder::GraphOptimizationLevel},
  value::TensorRef,
};
use smol_str::SmolStr;
use soundevents_dataset::RatedSoundEvent;
use std::{
  cmp::{Ordering, Reverse},
  collections::BinaryHeap,
  path::{Path, PathBuf},
};

/// The expected input sample rate for CED models.
pub const SAMPLE_RATE_HZ: usize = 16_000;

/// The default window size used by the chunked inference helpers: 10 seconds at 16 kHz.
pub const DEFAULT_CHUNK_SAMPLES: usize = 160_000;

/// Number of model output classes.
pub const NUM_CLASSES: usize = RatedSoundEvent::events().len();

#[cfg(feature = "bundled-tiny")]
const BUNDLED_TINY_MODEL: &[u8] =
  include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/tiny.onnx"));

/// Options for constructing a [`Classifier`] from an ONNX model on disk.
#[derive(Debug, Clone)]
pub struct Options {
  model_path: Option<PathBuf>,
  optimization_level: GraphOptimizationLevel,
}

impl Default for Options {
  fn default() -> Self {
    Self {
      model_path: None,
      optimization_level: GraphOptimizationLevel::Disable,
    }
  }
}

impl Options {
  /// Creates options pointing to the given ONNX model file.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(model_path: impl Into<PathBuf>) -> Self {
    Self::default().with_model_path(model_path)
  }

  /// Returns the model path, if one has been configured.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn model_path(&self) -> Option<&PathBuf> {
    self.model_path.as_ref()
  }

  /// Returns the optimization level for the ONNX Runtime session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
  }

  /// Sets the model path.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
    self.set_model_path(model_path);
    self
  }

  /// Sets the model path.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_model_path(&mut self, model_path: impl Into<PathBuf>) -> &mut Self {
    self.model_path = Some(model_path.into());
    self
  }

  /// Clears the configured model path.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn clear_model_path(&mut self) -> &mut Self {
    self.model_path = None;
    self
  }

  /// Sets the optimization level for the ONNX Runtime session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_optimization_level(mut self, level: GraphOptimizationLevel) -> Self {
    self.set_optimization_level(level);
    self
  }

  /// Sets the optimization level for the ONNX Runtime session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_optimization_level(&mut self, level: GraphOptimizationLevel) -> &mut Self {
    self.optimization_level = level;
    self
  }
}

/// Controls how chunked inference aggregates chunk confidences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkAggregation {
  /// Average confidence across all chunks.
  Mean,
  /// Keep the peak confidence seen in any chunk.
  Max,
}

/// Options for chunked inference over long clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkingOptions {
  window_samples: usize,
  hop_samples: usize,
  batch_size: usize,
  aggregation: ChunkAggregation,
}

impl Default for ChunkingOptions {
  fn default() -> Self {
    Self {
      window_samples: DEFAULT_CHUNK_SAMPLES,
      hop_samples: DEFAULT_CHUNK_SAMPLES,
      batch_size: 1,
      aggregation: ChunkAggregation::Mean,
    }
  }
}

impl ChunkingOptions {
  /// Returns the chunk window size in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn window_samples(&self) -> usize {
    self.window_samples
  }

  /// Returns the chunk hop size in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn hop_samples(&self) -> usize {
    self.hop_samples
  }

  /// Returns the maximum number of equal-length chunks to batch into one model call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn batch_size(&self) -> usize {
    self.batch_size
  }

  /// Returns the aggregation strategy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aggregation(&self) -> ChunkAggregation {
    self.aggregation
  }

  /// Sets the chunk window size in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_window_samples(mut self, window_samples: usize) -> Self {
    self.window_samples = window_samples;
    self
  }

  /// Sets the chunk hop size in samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_hop_samples(mut self, hop_samples: usize) -> Self {
    self.hop_samples = hop_samples;
    self
  }

  /// Sets the chunk batch size used by batched chunked inference.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
    self.batch_size = batch_size;
    self
  }

  /// Sets the aggregation strategy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_aggregation(mut self, aggregation: ChunkAggregation) -> Self {
    self.aggregation = aggregation;
    self
  }
}

/// Errors from [`Classifier`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
  /// ONNX Runtime error.
  #[error(transparent)]
  Ort(#[from] ort::Error),
  /// No model path was provided for file-based loading.
  #[error("a model path is required when loading from file")]
  MissingModelPath,
  /// The model exposes no input tensor.
  #[error("model exposes no input tensor")]
  MissingInputTensor,
  /// The model exposes no output tensor.
  #[error("model exposes no output tensor")]
  MissingOutputTensor,
  /// Empty audio was passed to the classifier.
  #[error("input audio is empty; expected mono {SAMPLE_RATE_HZ} Hz samples")]
  EmptyInput,
  /// An empty batch was passed to the classifier.
  #[error("input batch is empty; expected at least one mono {SAMPLE_RATE_HZ} Hz clip")]
  EmptyBatch,
  /// Model output is empty.
  #[error("model returned empty output")]
  EmptyOutput,
  /// The model returned an unexpected output shape.
  #[error(
    "unexpected model output shape {shape:?}; expected batch scores for {expected_batch} x {expected_classes}"
  )]
  UnexpectedOutputShape {
    /// The expected batch size.
    expected_batch: usize,
    /// The expected number of classes.
    expected_classes: usize,
    /// The actual output shape returned by the model.
    shape: Vec<i64>,
  },
  /// The model returned a class count that does not match the rated label set.
  #[error("model returned {actual} classes, expected {expected}")]
  UnexpectedClassCount {
    /// The expected number of classes.
    expected: usize,
    /// The actual number of classes returned by the model.
    actual: usize,
  },
  /// Batch members must have the same length to be packed into one tensor.
  #[error(
    "batched inference requires equal clip lengths; expected {expected} samples, got {actual}"
  )]
  MismatchedBatchLength {
    /// The clip length established by the first batch member.
    expected: usize,
    /// The mismatched clip length encountered later in the batch.
    actual: usize,
  },
  /// A model class index could not be resolved to a rated entry.
  #[error("no rated sound event exists for model class index {index}")]
  MissingRatedEventIndex {
    /// The model output class index that could not be resolved.
    index: usize,
  },
  /// Invalid chunking parameters were provided.
  #[error(
    "chunking options require non-zero window, hop, and batch sizes (window={window_samples}, hop={hop_samples}, batch={batch_size})"
  )]
  InvalidChunkingOptions {
    /// The chunk window size in samples.
    window_samples: usize,
    /// The chunk hop size in samples.
    hop_samples: usize,
    /// The chunk batch size.
    batch_size: usize,
  },
}

/// A single classification result with both model-space and ontology-space metadata.
#[derive(Debug, Clone)]
pub struct EventPrediction {
  event: &'static RatedSoundEvent,
  confidence: f32,
}

impl EventPrediction {
  /// Model output class index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn index(&self) -> usize {
    self.event().index()
  }

  /// The resolved rated AudioSet event.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn event(&self) -> &'static RatedSoundEvent {
    self.event
  }

  /// Canonical AudioSet display name for this class.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'static str {
    self.event().name()
  }

  /// Stable AudioSet identifier such as `"/m/09x0r"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn id(&self) -> &'static str {
    self.event().id()
  }

  /// Confidence after applying a sigmoid to the model output.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn confidence(&self) -> f32 {
    self.confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn from_confidence(class_index: usize, confidence: f32) -> Result<Self, ClassifierError> {
    let event = RatedSoundEvent::from_index(class_index)
      .ok_or(ClassifierError::MissingRatedEventIndex { index: class_index })?;

    Ok(Self { event, confidence })
  }
}

#[derive(Debug, Clone, Copy)]
struct RankedScore {
  class_index: usize,
  score: f32,
}

impl PartialEq for RankedScore {
  fn eq(&self, other: &Self) -> bool {
    self.class_index == other.class_index && self.score.total_cmp(&other.score) == Ordering::Equal
  }
}

impl Eq for RankedScore {}

impl PartialOrd for RankedScore {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for RankedScore {
  fn cmp(&self, other: &Self) -> Ordering {
    self
      .score
      .total_cmp(&other.score)
      .then_with(|| other.class_index.cmp(&self.class_index))
  }
}

/// CED sound event classifier.
pub struct Classifier {
  session: Session,
  input_name: SmolStr,
  output_name: SmolStr,
  input_scratch: Vec<f32>,
  confidence_scratch: Vec<f32>,
}

impl Classifier {
  /// Load a CED ONNX model from disk.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(opts: Options) -> Result<Self, ClassifierError> {
    let model_path = opts.model_path().ok_or(ClassifierError::MissingModelPath)?;
    Self::from_file_with_optimization(model_path, opts.optimization_level())
  }

  /// Load a CED ONNX model from disk with default optimization settings.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, ClassifierError> {
    Self::from_file_with_optimization(model_path, GraphOptimizationLevel::Disable)
  }

  /// Load a CED ONNX model directly from in-memory bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn from_memory(
    model_bytes: &[u8],
    optimization_level: GraphOptimizationLevel,
  ) -> Result<Self, ClassifierError> {
    let session = Session::builder()?
      .with_optimization_level(optimization_level)
      .map_err(ort::Error::from)?
      .commit_from_memory(model_bytes)?;

    Self::from_session(session)
  }

  /// Load the bundled CED-tiny model from the crate package.
  #[cfg(feature = "bundled-tiny")]
  #[cfg_attr(docsrs, doc(cfg(feature = "bundled-tiny")))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn tiny(opts: Options) -> Result<Self, ClassifierError> {
    Self::from_memory(BUNDLED_TINY_MODEL, opts.optimization_level())
  }

  /// Run the model on a mono 16 kHz clip and return the raw output scores.
  ///
  /// The clip is passed through at its original duration without truncation
  /// or repeat-padding.
  pub fn predict_raw_scores(&mut self, samples_16k: &[f32]) -> Result<Vec<f32>, ClassifierError> {
    self.with_raw_scores(samples_16k, |raw_scores| Ok(raw_scores.to_vec()))
  }

  /// Run the model on a batch of equal-length mono 16 kHz clips.
  ///
  /// Every clip in `batch_16k` must be non-empty and have the same number of
  /// samples. This low-level API is intended for service-layer batching and
  /// for chunked inference over fixed windows. If you need a single
  /// row-major buffer instead of one allocation per clip, prefer
  /// [`predict_raw_scores_batch_flat`](Self::predict_raw_scores_batch_flat) or
  /// [`predict_raw_scores_batch_into`](Self::predict_raw_scores_batch_into).
  pub fn predict_raw_scores_batch(
    &mut self,
    batch_16k: &[&[f32]],
  ) -> Result<Vec<Vec<f32>>, ClassifierError> {
    self.with_raw_scores_batch(batch_16k, |raw_scores, batch_size| {
      Ok(
        raw_scores
          .chunks_exact(NUM_CLASSES)
          .take(batch_size)
          .map(|scores| scores.to_vec())
          .collect(),
      )
    })
  }

  /// Run the model on a batch of equal-length mono 16 kHz clips and return
  /// all raw scores in one row-major buffer.
  ///
  /// The returned vector contains `batch_16k.len() * NUM_CLASSES` elements in
  /// `[batch, class]` order, so callers can iterate with
  /// `.chunks_exact(NUM_CLASSES)`.
  pub fn predict_raw_scores_batch_flat(
    &mut self,
    batch_16k: &[&[f32]],
  ) -> Result<Vec<f32>, ClassifierError> {
    self.with_raw_scores_batch(batch_16k, |raw_scores, _| Ok(raw_scores.to_vec()))
  }

  /// Run the model on a batch of equal-length mono 16 kHz clips and write the
  /// raw scores into a caller-provided row-major buffer.
  ///
  /// `out` is cleared before writing and then filled with
  /// `batch_16k.len() * NUM_CLASSES` elements in `[batch, class]` order.
  pub fn predict_raw_scores_batch_into(
    &mut self,
    batch_16k: &[&[f32]],
    out: &mut Vec<f32>,
  ) -> Result<(), ClassifierError> {
    self.with_raw_scores_batch(batch_16k, |raw_scores, _| {
      out.clear();
      out.reserve(raw_scores.len());
      out.extend_from_slice(raw_scores);
      Ok(())
    })
  }

  fn with_raw_scores<T>(
    &mut self,
    samples_16k: &[f32],
    f: impl FnOnce(&[f32]) -> Result<T, ClassifierError>,
  ) -> Result<T, ClassifierError> {
    self.with_raw_scores_batch(&[samples_16k], |raw_scores, _| f(raw_scores))
  }

  fn with_raw_scores_batch<T>(
    &mut self,
    batch_16k: &[&[f32]],
    f: impl FnOnce(&[f32], usize) -> Result<T, ClassifierError>,
  ) -> Result<T, ClassifierError> {
    let chunk_len = validate_batch_inputs(batch_16k)?;
    let batch_size = batch_16k.len();
    let total_samples = batch_size * chunk_len;

    self.input_scratch.clear();
    self.input_scratch.reserve(total_samples);
    for clip in batch_16k {
      self.input_scratch.extend_from_slice(clip);
    }

    let input_ref =
      TensorRef::from_array_view(([batch_size, chunk_len], self.input_scratch.as_slice()))?;
    let outputs = self
      .session
      .run(ort::inputs![self.input_name.as_str() => input_ref])?;
    let (shape, raw_scores) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;

    validate_output(shape, raw_scores, batch_size)?;

    f(raw_scores, batch_size)
  }

  /// Classify a batch of equal-length mono 16 kHz clips and return every class in model order.
  pub fn classify_all_batch(
    &mut self,
    batch_16k: &[&[f32]],
  ) -> Result<Vec<Vec<EventPrediction>>, ClassifierError> {
    self.with_raw_scores_batch(batch_16k, |raw_scores, batch_size| {
      raw_scores
        .chunks_exact(NUM_CLASSES)
        .take(batch_size)
        .map(|row| {
          row
            .iter()
            .copied()
            .enumerate()
            .map(|(class_index, raw_score)| {
              EventPrediction::from_confidence(class_index, sigmoid(raw_score))
            })
            .collect()
        })
        .collect()
    })
  }

  /// Classify a mono 16 kHz clip and return every class in model order.
  pub fn classify_all(
    &mut self,
    samples_16k: &[f32],
  ) -> Result<Vec<EventPrediction>, ClassifierError> {
    self.with_raw_scores(samples_16k, |raw_scores| {
      raw_scores
        .iter()
        .copied()
        .enumerate()
        .map(|(class_index, raw_score)| {
          EventPrediction::from_confidence(class_index, sigmoid(raw_score))
        })
        .collect()
    })
  }

  /// Classify a mono 16 kHz clip and return the top `k` classes.
  pub fn classify(
    &mut self,
    samples_16k: &[f32],
    top_k: usize,
  ) -> Result<Vec<EventPrediction>, ClassifierError> {
    ensure_non_empty(samples_16k)?;

    if top_k == 0 {
      return Ok(Vec::new());
    }

    self.with_raw_scores(samples_16k, |raw_scores| {
      top_k_from_scores(raw_scores.iter().copied().enumerate(), top_k, sigmoid)
    })
  }

  /// Classify a batch of equal-length mono 16 kHz clips and return the top `k` classes for each clip.
  pub fn classify_batch(
    &mut self,
    batch_16k: &[&[f32]],
    top_k: usize,
  ) -> Result<Vec<Vec<EventPrediction>>, ClassifierError> {
    let _ = validate_batch_inputs(batch_16k)?;

    if top_k == 0 {
      return Ok((0..batch_16k.len()).map(|_| Vec::new()).collect());
    }

    self.with_raw_scores_batch(batch_16k, |raw_scores, batch_size| {
      raw_scores
        .chunks_exact(NUM_CLASSES)
        .take(batch_size)
        .map(|row| top_k_from_scores(row.iter().copied().enumerate(), top_k, sigmoid))
        .collect()
    })
  }

  /// Classify a long clip by chunking it into windows and aggregating chunk confidences.
  pub fn classify_all_chunked(
    &mut self,
    samples_16k: &[f32],
    options: ChunkingOptions,
  ) -> Result<Vec<EventPrediction>, ClassifierError> {
    self.with_aggregated_confidences(samples_16k, options, |confidences| {
      confidences
        .iter()
        .copied()
        .enumerate()
        .map(|(class_index, confidence)| EventPrediction::from_confidence(class_index, confidence))
        .collect()
    })
  }

  /// Chunk a long clip, aggregate chunk confidences, and return the top `k` classes.
  pub fn classify_chunked(
    &mut self,
    samples_16k: &[f32],
    top_k: usize,
    options: ChunkingOptions,
  ) -> Result<Vec<EventPrediction>, ClassifierError> {
    ensure_non_empty(samples_16k)?;

    if top_k == 0 {
      return Ok(Vec::new());
    }

    self.with_aggregated_confidences(samples_16k, options, |confidences| {
      top_k_from_scores(
        confidences.iter().copied().enumerate(),
        top_k,
        |confidence| confidence,
      )
    })
  }

  fn from_file_with_optimization(
    model_path: impl AsRef<Path>,
    optimization_level: GraphOptimizationLevel,
  ) -> Result<Self, ClassifierError> {
    let session = Session::builder()?
      .with_optimization_level(optimization_level)
      .map_err(ort::Error::from)?
      .commit_from_file(model_path.as_ref())?;

    Self::from_session(session)
  }

  fn from_session(session: Session) -> Result<Self, ClassifierError> {
    let input_name = SmolStr::new(
      session
        .inputs()
        .first()
        .ok_or(ClassifierError::MissingInputTensor)?
        .name(),
    );
    let output_name = SmolStr::new(
      session
        .outputs()
        .first()
        .ok_or(ClassifierError::MissingOutputTensor)?
        .name(),
    );

    Ok(Self {
      session,
      input_name,
      output_name,
      input_scratch: Vec::new(),
      confidence_scratch: Vec::with_capacity(NUM_CLASSES),
    })
  }

  fn with_aggregated_confidences<T>(
    &mut self,
    samples_16k: &[f32],
    options: ChunkingOptions,
    f: impl FnOnce(&[f32]) -> Result<T, ClassifierError>,
  ) -> Result<T, ClassifierError> {
    let mut confidences = std::mem::take(&mut self.confidence_scratch);
    let result = fill_aggregated_confidences(self, &mut confidences, samples_16k, options)
      .and_then(|()| f(&confidences));
    confidences.clear();
    self.confidence_scratch = confidences;
    result
  }
}

fn fill_aggregated_confidences(
  classifier: &mut Classifier,
  aggregated: &mut Vec<f32>,
  samples_16k: &[f32],
  options: ChunkingOptions,
) -> Result<(), ClassifierError> {
  ensure_non_empty(samples_16k)?;
  validate_chunking(options)?;

  let mut chunk_count = 0usize;
  let mut initialized = false;

  for batch in chunk_batches(samples_16k, options) {
    accumulate_chunk_batch(
      classifier,
      aggregated,
      &batch,
      options.aggregation(),
      initialized,
    )?;
    chunk_count += batch.len();
    initialized = true;
  }

  if matches!(options.aggregation(), ChunkAggregation::Mean) && chunk_count > 1 {
    let denominator = chunk_count as f32;
    for aggregate in aggregated.iter_mut() {
      *aggregate /= denominator;
    }
  }

  Ok(())
}

fn accumulate_chunk_batch(
  classifier: &mut Classifier,
  aggregated: &mut Vec<f32>,
  batch: &[&[f32]],
  aggregation: ChunkAggregation,
  initialized: bool,
) -> Result<(), ClassifierError> {
  classifier.with_raw_scores_batch(batch, |raw_scores, batch_size| {
    for (row_index, row) in raw_scores
      .chunks_exact(NUM_CLASSES)
      .take(batch_size)
      .enumerate()
    {
      if !initialized && row_index == 0 {
        aggregated.clear();
        aggregated.extend(row.iter().copied().map(sigmoid));
        continue;
      }

      match aggregation {
        ChunkAggregation::Mean => {
          for (aggregate, raw_score) in aggregated.iter_mut().zip(row.iter().copied()) {
            *aggregate += sigmoid(raw_score);
          }
        }
        ChunkAggregation::Max => {
          for (aggregate, raw_score) in aggregated.iter_mut().zip(row.iter().copied()) {
            *aggregate = aggregate.max(sigmoid(raw_score));
          }
        }
      }
    }

    Ok(())
  })
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn top_k_from_scores(
  scores: impl IntoIterator<Item = (usize, f32)>,
  top_k: usize,
  map_score: impl Fn(f32) -> f32,
) -> Result<Vec<EventPrediction>, ClassifierError> {
  if top_k == 0 {
    return Ok(Vec::new());
  }

  let mut heap = BinaryHeap::with_capacity(top_k);

  for (class_index, score) in scores {
    let ranked = RankedScore { class_index, score };
    let candidate = Reverse(ranked);

    if heap.len() < top_k {
      heap.push(candidate);
      continue;
    }

    if heap.peek().is_some_and(|smallest| candidate.0 > smallest.0) {
      heap.pop();
      heap.push(candidate);
    }
  }

  let mut predictions = Vec::with_capacity(heap.len());
  while let Some(entry) = heap.pop() {
    let ranked = entry.0;
    predictions.push(EventPrediction::from_confidence(
      ranked.class_index,
      map_score(ranked.score),
    )?);
  }
  predictions.reverse();
  Ok(predictions)
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn ensure_non_empty(samples_16k: &[f32]) -> Result<(), ClassifierError> {
  if samples_16k.is_empty() {
    return Err(ClassifierError::EmptyInput);
  }

  Ok(())
}

fn validate_chunking(options: ChunkingOptions) -> Result<(), ClassifierError> {
  if options.window_samples() == 0 || options.hop_samples() == 0 || options.batch_size() == 0 {
    return Err(ClassifierError::InvalidChunkingOptions {
      window_samples: options.window_samples(),
      hop_samples: options.hop_samples(),
      batch_size: options.batch_size(),
    });
  }

  Ok(())
}

fn validate_output(
  shape: &ort::value::Shape,
  raw_scores: &[f32],
  expected_batch_size: usize,
) -> Result<(), ClassifierError> {
  if raw_scores.is_empty() {
    return Err(ClassifierError::EmptyOutput);
  }

  let expected_values = expected_batch_size.saturating_mul(NUM_CLASSES);
  if raw_scores.len() != expected_values {
    if raw_scores.len() % expected_batch_size.max(1) == 0 {
      return Err(ClassifierError::UnexpectedClassCount {
        expected: NUM_CLASSES,
        actual: raw_scores.len() / expected_batch_size.max(1),
      });
    }

    return Err(ClassifierError::UnexpectedOutputShape {
      expected_batch: expected_batch_size,
      expected_classes: NUM_CLASSES,
      shape: shape.to_vec(),
    });
  }

  // Keep shape validation strict: released CED exports use `[batch, 527]`
  // (and some runtimes collapse batch-one to `[527]`). Accepting arbitrary
  // shapes with the same element count would make tensor-packing bugs much
  // harder to catch.
  match &shape[..] {
    [classes] if expected_batch_size == 1 && *classes as usize == NUM_CLASSES => Ok(()),
    [batch, classes]
      if *batch as usize == expected_batch_size && *classes as usize == NUM_CLASSES =>
    {
      Ok(())
    }
    _ => Err(ClassifierError::UnexpectedOutputShape {
      expected_batch: expected_batch_size,
      expected_classes: NUM_CLASSES,
      shape: shape.to_vec(),
    }),
  }
}

/// Validates a batch of clips and returns the common clip length in samples.
fn validate_batch_inputs(batch_16k: &[&[f32]]) -> Result<usize, ClassifierError> {
  let Some(first) = batch_16k.first() else {
    return Err(ClassifierError::EmptyBatch);
  };

  ensure_non_empty(first)?;
  let expected = first.len();

  for clip in &batch_16k[1..] {
    ensure_non_empty(clip)?;
    if clip.len() != expected {
      return Err(ClassifierError::MismatchedBatchLength {
        expected,
        actual: clip.len(),
      });
    }
  }

  Ok(expected)
}

/// Groups consecutive equal-length chunks into batches so one tensor never
/// mixes the usual short tail chunk with full-size windows.
fn chunk_batches(samples: &[f32], options: ChunkingOptions) -> impl Iterator<Item = Vec<&[f32]>> {
  let mut chunks =
    chunk_slices(samples, options.window_samples(), options.hop_samples()).peekable();

  std::iter::from_fn(move || {
    let first = chunks.next()?;
    let first_len = first.len();
    let mut batch = Vec::with_capacity(options.batch_size());
    batch.push(first);

    while batch.len() < options.batch_size() {
      let Some(next_len) = chunks.peek().map(|chunk| chunk.len()) else {
        break;
      };
      if next_len != first_len {
        break;
      }
      batch.push(chunks.next().expect("peeked chunk must exist"));
    }

    Some(batch)
  })
}

fn chunk_slices(
  samples: &[f32],
  window_samples: usize,
  hop_samples: usize,
) -> impl Iterator<Item = &[f32]> {
  let mut start = 0usize;

  std::iter::from_fn(move || {
    if start >= samples.len() {
      return None;
    }

    let end = start.saturating_add(window_samples).min(samples.len());
    let chunk = &samples[start..end];
    start = start.saturating_add(hop_samples);
    Some(chunk)
  })
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[cfg(feature = "bundled-tiny")]
  fn pseudo_audio(len: usize, mut seed: u64) -> Vec<f32> {
    let mut samples = Vec::with_capacity(len);
    for _ in 0..len {
      seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
      let value = ((seed >> 40) as u32) as f32 / u32::MAX as f32;
      samples.push(value * 2.0 - 1.0);
    }
    samples
  }

  #[test]
  fn rated_indices_round_trip() {
    for event in RatedSoundEvent::events() {
      assert_eq!(
        RatedSoundEvent::from_index(event.index()).map(RatedSoundEvent::id),
        Some(event.id())
      );
    }
  }

  #[test]
  fn chunk_iterator_keeps_tail_chunk_without_padding() {
    let samples = [0.0_f32; 12];
    let chunk_lengths = chunk_slices(&samples, 5, 4)
      .map(|chunk| chunk.len())
      .collect::<Vec<_>>();

    assert_eq!(chunk_lengths, vec![5, 5, 4]);
  }

  #[test]
  fn default_chunking_matches_ced_window_size() {
    let options = ChunkingOptions::default();

    assert_eq!(options.window_samples(), DEFAULT_CHUNK_SAMPLES);
    assert_eq!(options.hop_samples(), DEFAULT_CHUNK_SAMPLES);
    assert_eq!(options.batch_size(), 1);
    assert_eq!(options.aggregation(), ChunkAggregation::Mean);
  }

  #[test]
  fn chunking_options_can_configure_batch_size() {
    let options = ChunkingOptions::default().with_batch_size(8);
    assert_eq!(options.batch_size(), 8);
  }

  #[test]
  fn validate_batch_inputs_requires_equal_non_empty_clips() {
    assert!(matches!(
      validate_batch_inputs(&[]),
      Err(ClassifierError::EmptyBatch)
    ));
    assert!(matches!(
      validate_batch_inputs(&[&[]]),
      Err(ClassifierError::EmptyInput)
    ));
    assert!(matches!(
      validate_batch_inputs(&[&[0.0, 1.0], &[0.0]]),
      Err(ClassifierError::MismatchedBatchLength {
        expected: 2,
        actual: 1,
      })
    ));
  }

  #[cfg(feature = "bundled-tiny")]
  #[test]
  fn batched_predict_raw_scores_matches_sequential_inference() {
    let clip_a = pseudo_audio(SAMPLE_RATE_HZ * 2, 0x1234_5678);
    let clip_b = pseudo_audio(SAMPLE_RATE_HZ * 2, 0x9abc_def0);

    let mut sequential = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let seq_a = sequential
      .predict_raw_scores(&clip_a)
      .expect("sequential clip a");
    let seq_b = sequential
      .predict_raw_scores(&clip_b)
      .expect("sequential clip b");

    let mut batched = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let batch = batched
      .predict_raw_scores_batch(&[&clip_a, &clip_b])
      .expect("batched inference");

    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].len(), seq_a.len());
    assert_eq!(batch[1].len(), seq_b.len());

    for (expected, actual) in seq_a.iter().zip(batch[0].iter()) {
      assert!((expected - actual).abs() < 1e-6);
    }
    for (expected, actual) in seq_b.iter().zip(batch[1].iter()) {
      assert!((expected - actual).abs() < 1e-6);
    }
  }

  #[cfg(feature = "bundled-tiny")]
  #[test]
  fn flat_and_into_batch_raw_scores_match_sequential_inference() {
    let clip_a = pseudo_audio(SAMPLE_RATE_HZ * 2, 0x1357_9bdf);
    let clip_b = pseudo_audio(SAMPLE_RATE_HZ * 2, 0x2468_ace0);

    let mut sequential = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let seq_a = sequential
      .predict_raw_scores(&clip_a)
      .expect("sequential clip a");
    let seq_b = sequential
      .predict_raw_scores(&clip_b)
      .expect("sequential clip b");

    let mut batched = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let flat = batched
      .predict_raw_scores_batch_flat(&[&clip_a, &clip_b])
      .expect("flat batched inference");

    assert_eq!(flat.len(), 2 * NUM_CLASSES);
    for (expected, actual) in seq_a.iter().zip(flat[..NUM_CLASSES].iter()) {
      assert!((expected - actual).abs() < 1e-6);
    }
    for (expected, actual) in seq_b.iter().zip(flat[NUM_CLASSES..].iter()) {
      assert!((expected - actual).abs() < 1e-6);
    }

    let mut into = vec![1.0; 7];
    batched
      .predict_raw_scores_batch_into(&[&clip_a, &clip_b], &mut into)
      .expect("into batched inference");
    assert_eq!(into, flat);
  }

  #[cfg(feature = "bundled-tiny")]
  #[test]
  fn chunked_batching_matches_batch_size_one() {
    let clip = pseudo_audio(DEFAULT_CHUNK_SAMPLES * 2 + 40_000, 0x0ddc_0ffe);
    let single_opts = ChunkingOptions::default()
      .with_hop_samples(DEFAULT_CHUNK_SAMPLES / 2)
      .with_batch_size(1);
    let batched_opts = single_opts.with_batch_size(4);

    let mut single = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let single_predictions = single
      .classify_all_chunked(&clip, single_opts)
      .expect("chunked single-batch inference");

    let mut batched = Classifier::tiny(Options::default()).expect("load bundled classifier");
    let batched_predictions = batched
      .classify_all_chunked(&clip, batched_opts)
      .expect("chunked batched inference");

    assert_eq!(single_predictions.len(), batched_predictions.len());
    for (expected, actual) in single_predictions.iter().zip(batched_predictions.iter()) {
      assert_eq!(expected.index(), actual.index());
      assert!((expected.confidence() - actual.confidence()).abs() < 1e-6);
    }
  }

  #[test]
  fn top_k_selection_returns_descending_predictions() {
    let predictions = top_k_from_scores(
      vec![0.0, 3.0, -1.0, 1.5].into_iter().enumerate(),
      2,
      sigmoid,
    )
    .unwrap();
    let indices = predictions
      .into_iter()
      .map(|prediction| prediction.index())
      .collect::<Vec<_>>();

    assert_eq!(indices, vec![1, 3]);
  }
}
