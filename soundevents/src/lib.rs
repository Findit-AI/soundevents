use ort::{
  session::{Session, builder::GraphOptimizationLevel},
  value::TensorRef,
};
use smol_str::SmolStr;
use soundevents_dataset::RatedSoundEvent;
use std::path::{Path, PathBuf};

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
  aggregation: ChunkAggregation,
}

impl Default for ChunkingOptions {
  fn default() -> Self {
    Self {
      window_samples: DEFAULT_CHUNK_SAMPLES,
      hop_samples: DEFAULT_CHUNK_SAMPLES,
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
  /// Model output is empty.
  #[error("model returned empty output")]
  EmptyOutput,
  /// The model returned an unexpected output shape.
  #[error(
    "unexpected model output shape {shape:?}; expected batch-one scores for {expected} classes"
  )]
  UnexpectedOutputShape { expected: usize, shape: Vec<i64> },
  /// The model returned a class count that does not match the rated label set.
  #[error("model returned {actual} classes, expected {expected}")]
  UnexpectedClassCount { expected: usize, actual: usize },
  /// Invalid chunking parameters were provided.
  #[error(
    "chunking options require non-zero window and hop sizes (window={window_samples}, hop={hop_samples})"
  )]
  InvalidChunkingOptions {
    window_samples: usize,
    hop_samples: usize,
  },
}

/// A single classification result with both model-space and ontology-space metadata.
#[derive(Debug, Clone)]
pub struct TagConfidence {
  event: &'static RatedSoundEvent,
  confidence: f32,
}

impl TagConfidence {
  /// Model output class index.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn class_index(&self) -> usize {
    self.event.index()
  }

  /// Alias for [`TagConfidence::class_index`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn index(&self) -> usize {
    self.class_index()
  }

  /// The resolved rated AudioSet event.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn event(&self) -> &'static RatedSoundEvent {
    self.event
  }

  /// Canonical AudioSet display name for this class.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn display_name(&self) -> &'static str {
    self.event.name()
  }

  /// Alias for [`TagConfidence::display_name`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn label(&self) -> &'static str {
    self.display_name()
  }

  /// Stable AudioSet identifier such as `"/m/09x0r"`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn audioset_id(&self) -> &'static str {
    self.event.id()
  }

  /// Confidence after applying a sigmoid to the model output.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn confidence(&self) -> f32 {
    self.confidence
  }

  fn from_confidence(class_index: usize, confidence: f32) -> Self {
    let event = RatedSoundEvent::from_index(class_index)
      .unwrap_or_else(|| panic!("missing RatedSoundEvent for model class index {class_index}"));

    Self { event, confidence }
  }
}

/// CED sound event classifier.
pub struct Classifier {
  session: Session,
  input_name: SmolStr,
  output_name: SmolStr,
}

impl Classifier {
  /// Load a CED ONNX model from disk.
  pub fn new(opts: Options) -> Result<Self, ClassifierError> {
    let model_path = opts.model_path().ok_or(ClassifierError::MissingModelPath)?;
    Self::from_file_with_optimization(model_path, opts.optimization_level())
  }

  /// Load a CED ONNX model from disk with default optimization settings.
  pub fn from_file(model_path: impl AsRef<Path>) -> Result<Self, ClassifierError> {
    Self::from_file_with_optimization(model_path, GraphOptimizationLevel::Disable)
  }

  /// Load a CED ONNX model directly from in-memory bytes.
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
  pub fn bundled_tiny() -> Result<Self, ClassifierError> {
    Self::from_memory(BUNDLED_TINY_MODEL, GraphOptimizationLevel::Disable)
  }

  /// Run the model on a mono 16 kHz clip and return the raw output scores.
  ///
  /// The clip is passed through at its original duration without truncation
  /// or repeat-padding.
  pub fn predict_raw_scores(&mut self, samples_16k: &[f32]) -> Result<Vec<f32>, ClassifierError> {
    ensure_non_empty(samples_16k)?;

    let input_ref = TensorRef::from_array_view(([1usize, samples_16k.len()], samples_16k))?;
    let outputs = self
      .session
      .run(ort::inputs![self.input_name.as_str() => input_ref])?;
    let (shape, raw_scores) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;

    validate_output(shape, raw_scores)?;

    Ok(raw_scores.to_vec())
  }

  /// Classify a mono 16 kHz clip and return every class in model order.
  pub fn classify_all(
    &mut self,
    samples_16k: &[f32],
  ) -> Result<Vec<TagConfidence>, ClassifierError> {
    Ok(
      self
        .predict_raw_scores(samples_16k)?
        .into_iter()
        .enumerate()
        .map(|(class_index, raw_score)| {
          TagConfidence::from_confidence(class_index, sigmoid(raw_score))
        })
        .collect(),
    )
  }

  /// Classify a mono 16 kHz clip and return the top `k` classes.
  pub fn classify(
    &mut self,
    samples_16k: &[f32],
    top_k: usize,
  ) -> Result<Vec<TagConfidence>, ClassifierError> {
    ensure_non_empty(samples_16k)?;

    if top_k == 0 {
      return Ok(Vec::new());
    }

    Ok(top_k_from_raw_scores(
      self.predict_raw_scores(samples_16k)?,
      top_k,
    ))
  }

  /// Classify a long clip by chunking it into windows and aggregating chunk confidences.
  pub fn classify_all_chunked(
    &mut self,
    samples_16k: &[f32],
    options: ChunkingOptions,
  ) -> Result<Vec<TagConfidence>, ClassifierError> {
    Ok(
      aggregate_chunk_confidences(self, samples_16k, options)?
        .into_iter()
        .enumerate()
        .map(|(class_index, confidence)| TagConfidence::from_confidence(class_index, confidence))
        .collect(),
    )
  }

  /// Chunk a long clip, aggregate chunk confidences, and return the top `k` classes.
  pub fn classify_chunked(
    &mut self,
    samples_16k: &[f32],
    top_k: usize,
    options: ChunkingOptions,
  ) -> Result<Vec<TagConfidence>, ClassifierError> {
    ensure_non_empty(samples_16k)?;

    if top_k == 0 {
      return Ok(Vec::new());
    }

    Ok(top_k_from_confidences(
      aggregate_chunk_confidences(self, samples_16k, options)?,
      top_k,
    ))
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
    })
  }
}

fn aggregate_chunk_confidences(
  classifier: &mut Classifier,
  samples_16k: &[f32],
  options: ChunkingOptions,
) -> Result<Vec<f32>, ClassifierError> {
  ensure_non_empty(samples_16k)?;
  validate_chunking(options)?;

  let mut chunks = chunk_slices(samples_16k, options.window_samples(), options.hop_samples());
  let first_chunk = chunks
    .next()
    .expect("non-empty input should yield one chunk");
  let mut aggregated = classifier
    .predict_raw_scores(first_chunk)?
    .into_iter()
    .map(sigmoid)
    .collect::<Vec<_>>();
  let mut chunk_count = 1usize;

  for chunk in chunks {
    let chunk_confidences = classifier
      .predict_raw_scores(chunk)?
      .into_iter()
      .map(sigmoid)
      .collect::<Vec<_>>();

    match options.aggregation() {
      ChunkAggregation::Mean => {
        for (aggregate, confidence) in aggregated.iter_mut().zip(chunk_confidences) {
          *aggregate += confidence;
        }
      }
      ChunkAggregation::Max => {
        for (aggregate, confidence) in aggregated.iter_mut().zip(chunk_confidences) {
          *aggregate = aggregate.max(confidence);
        }
      }
    }

    chunk_count += 1;
  }

  if matches!(options.aggregation(), ChunkAggregation::Mean) && chunk_count > 1 {
    let denominator = chunk_count as f32;
    for aggregate in &mut aggregated {
      *aggregate /= denominator;
    }
  }

  Ok(aggregated)
}

fn top_k_from_raw_scores(raw_scores: Vec<f32>, top_k: usize) -> Vec<TagConfidence> {
  let mut ranked = raw_scores.into_iter().enumerate().collect::<Vec<_>>();
  ranked.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
  ranked.truncate(top_k.min(ranked.len()));

  ranked
    .into_iter()
    .map(|(class_index, raw_score)| TagConfidence::from_confidence(class_index, sigmoid(raw_score)))
    .collect()
}

fn top_k_from_confidences(confidences: Vec<f32>, top_k: usize) -> Vec<TagConfidence> {
  let mut ranked = confidences.into_iter().enumerate().collect::<Vec<_>>();
  ranked.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
  ranked.truncate(top_k.min(ranked.len()));

  ranked
    .into_iter()
    .map(|(class_index, confidence)| TagConfidence::from_confidence(class_index, confidence))
    .collect()
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn ensure_non_empty(samples_16k: &[f32]) -> Result<(), ClassifierError> {
  if samples_16k.is_empty() {
    return Err(ClassifierError::EmptyInput);
  }

  Ok(())
}

fn validate_chunking(options: ChunkingOptions) -> Result<(), ClassifierError> {
  if options.window_samples() == 0 || options.hop_samples() == 0 {
    return Err(ClassifierError::InvalidChunkingOptions {
      window_samples: options.window_samples(),
      hop_samples: options.hop_samples(),
    });
  }

  Ok(())
}

fn validate_output(shape: &ort::value::Shape, raw_scores: &[f32]) -> Result<(), ClassifierError> {
  if raw_scores.is_empty() {
    return Err(ClassifierError::EmptyOutput);
  }

  if raw_scores.len() != NUM_CLASSES {
    return Err(ClassifierError::UnexpectedClassCount {
      expected: NUM_CLASSES,
      actual: raw_scores.len(),
    });
  }

  match &shape[..] {
    [classes] if *classes as usize == NUM_CLASSES => Ok(()),
    [batch, classes] if *batch == 1 && *classes as usize == NUM_CLASSES => Ok(()),
    _ if shape.num_elements() == NUM_CLASSES => Ok(()),
    _ => Err(ClassifierError::UnexpectedOutputShape {
      expected: NUM_CLASSES,
      shape: shape.to_vec(),
    }),
  }
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

  #[test]
  fn rated_indices_round_trip() {
    for event in RatedSoundEvent::events() {
      let round_trip = RatedSoundEvent::from_index(event.index())
        .unwrap_or_else(|| panic!("failed to resolve rated index {}", event.index()));
      assert_eq!(round_trip.id(), event.id());
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
    assert_eq!(options.aggregation(), ChunkAggregation::Mean);
  }
}
