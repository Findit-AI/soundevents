//! The full AudioSet ontology — all 632 entries from upstream
//! [`ontology.json`](https://github.com/audioset/ontology), including
//! abstract container nodes and blacklisted classes.
//!
//! Use the [`rated`](crate::rated) module instead if you only need the
//! 527 classes from the released `class_labels_indices.csv`.

crate::define_sound_event! {
  /// A sound entry in the full AudioSet ontology.
  name: SoundEvent,
  /// Errors that can occur when looking up a [`SoundEvent`] by its code.
  error: UnknownSoundEventCode,
  error_message: "unknown sound event code: {0}",
}

mod generated;
