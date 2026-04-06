//! The released AudioSet rated label set — the 527 classes from
//! [`class_labels_indices.csv`](https://research.google.com/audioset/download.html)
//! that AudioSet annotators actually labeled on YouTube clips.
//!
//! This is a strict subset of the [`ontology`](crate::ontology) module:
//! abstract container nodes and blacklisted classes are excluded. The
//! children of a [`RatedSoundEvent`] reference only other rated entries,
//! so traversing the hierarchy stays inside the rated namespace.

crate::define_sound_event! {
  /// A sound entry in the rated AudioSet label set.
  name: RatedSoundEvent,
  /// Errors that can occur when looking up a [`RatedSoundEvent`] by its code.
  error: UnknownRatedSoundEventCode,
  error_message: "unknown rated sound event code: {0}",
}

mod generated;
pub use generated::*;
