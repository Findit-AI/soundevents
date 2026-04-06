use ort::{
  session::{Session, builder::GraphOptimizationLevel},
  value::TensorRef,
};
use smol_str::SmolStr;
use std::{
  path::{Path, PathBuf},
  sync::Mutex,
};

/// Fixed input length: 10 seconds at 16 kHz.
const WINDOW_SAMPLES: usize = 160_000;

/// Number of AudioSet classes.
const NUM_CLASSES: usize = 527;

/// Options for constructing a [`Classifier`].
#[derive(Debug, Clone)]
pub struct Options {
  model_path: PathBuf,
  optimization_level: GraphOptimizationLevel,
}

impl Options {
  /// Creates options pointing to the given ONNX model file.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(model_path: impl Into<PathBuf>) -> Self {
    Self {
      model_path: model_path.into(),
      optimization_level: GraphOptimizationLevel::Disable,
    }
  }

  /// Returns the model path.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn model_path(&self) -> &PathBuf {
    &self.model_path
  }

  /// Returns the optimization level for the ONNX Runtime session.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn optimization_level(&self) -> GraphOptimizationLevel {
    self.optimization_level
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

/// Errors from [`Classifier`] operations.
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
  /// ONNX Runtime error.
  #[error(transparent)]
  Ort(#[from] ort::Error),
  /// Tensor shape error.
  #[error(transparent)]
  Shape(#[from] ndarray::ShapeError),
  /// Model output is empty.
  #[error("model returned empty output")]
  EmptyOutput,
}

/// A single classification result: label index + sigmoid probability.
#[derive(Debug, Clone)]
pub struct TagConfidence {
  index: usize,
  confidence: f32,
}

impl TagConfidence {
  /// AudioSet class index (0–526).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn index(&self) -> usize {
    self.index
  }

  /// Sigmoid probability [0, 1].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn confidence(&self) -> f32 {
    self.confidence
  }
}

/// CED-tiny sound event classifier.
///
/// Thread-safe: the ONNX session is behind a [`Mutex`].
pub struct Classifier {
  session: Session,
  input_name: SmolStr,
  output_name: SmolStr,
}

impl Classifier {
  /// Load the CED-tiny ONNX model.
  pub fn new(opts: Options) -> Result<Self, ClassifierError> {
    let session = Session::builder()?
      .with_optimization_level(opts.optimization_level())
      .map_err(ort::Error::from)?
      .commit_from_file(&opts.model_path)?;

    let input_name = SmolStr::new(session.inputs()[0].name());
    let output_name = SmolStr::new(session.outputs()[0].name());

    Ok(Self {
      session,
      input_name,
      output_name,
    })
  }

  /// Classify a mono 16 kHz audio clip.
  ///
  /// Audio shorter than 10 s is repeat-padded; longer audio is truncated.
  /// Returns the top `k` classes sorted by descending confidence.
  pub fn classify(
    &mut self,
    samples_16k: &[f32],
    top_k: usize,
  ) -> Result<Vec<TagConfidence>, ClassifierError> {
    let input = Self::prepare_input(samples_16k);
    let input_arr = ndarray::Array2::from_shape_vec((1, WINDOW_SAMPLES), input)?;
    let input_ref = TensorRef::from_array_view(&input_arr)?;

    let outputs = self
      .session
      .run(ort::inputs![self.input_name.as_str() => input_ref])?;
    let (_, logits) = outputs[self.output_name.as_str()].try_extract_tensor::<f32>()?;

    if logits.is_empty() {
      return Err(ClassifierError::EmptyOutput);
    }

    let mut tags: Vec<TagConfidence> = logits
      .iter()
      .take(NUM_CLASSES)
      .enumerate()
      .map(|(i, &logit)| TagConfidence {
        index: i,
        confidence: sigmoid(logit),
      })
      .collect();

    tags.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    tags.truncate(top_k);

    Ok(tags)
  }

  /// Repeat-pad or truncate to exactly [`WINDOW_SAMPLES`].
  fn prepare_input(samples: &[f32]) -> Vec<f32> {
    if samples.len() >= WINDOW_SAMPLES {
      return samples[..WINDOW_SAMPLES].to_vec();
    }
    let mut out = Vec::with_capacity(WINDOW_SAMPLES);
    let repeats = WINDOW_SAMPLES / samples.len().max(1);
    for _ in 0..repeats {
      out.extend_from_slice(samples);
    }
    out.resize(WINDOW_SAMPLES, 0.0);
    out
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}
