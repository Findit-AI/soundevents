//! A template for creating Rust open-source repo on GitHub
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

pub use generated::*;

mod generated;

/// Errors that can occur when looking up a sound entry by name
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[error("unknown entry name: {0}")]
pub struct UnknownEntry<'a>(&'a str);

impl UnknownEntry<'_> {
  /// Get the name associated with the `UnknownEntry` error
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &str {
    self.0
  }
}

/// Errors that can occur when looking up a sound entry by code
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[error("unknown entry code: {0}")]
pub struct UnknownEntryCode(u64);

impl UnknownEntryCode {
  /// Get the code associated with the `UnknownEntryCode` error
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn code(&self) -> u64 {
    self.0
  }
}

/// A sound entry for the audioset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundEntry {
  code: u64,
  id: &'static str,
  name: &'static str,
  aliases: &'static [&'static str],
  description: &'static str,
  citation_uri: Option<&'static str>,
  children: &'static [&'static SoundEntry],
  restrictions: &'static [&'static str],
}

impl core::fmt::Display for SoundEntry {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{}", self.name)
  }
}

impl SoundEntry {
  /// Get the unique code for the sound entry, which is a hash of its name.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn encode(&self) -> u64 {
    self.code
  }

  /// Get the sound entry's id
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn id(&self) -> &'static str {
    self.id
  }

  /// Get the sound entry's name
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'static str {
    self.name
  }

  /// Get the sound entry's description
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn description(&self) -> &'static str {
    self.description
  }

  /// Get the sound entry's aliases
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn aliases(&self) -> &'static [&'static str] {
    self.aliases
  }

  /// Get the sound entry's citation url, if any
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn citation_uri(&self) -> Option<&'static str> {
    self.citation_uri
  }

  /// Get the sound entry's children sound entries
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn children(&self) -> &'static [&'static SoundEntry] {
    self.children
  }

  /// Get the sound entry's restrictions
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn restrictions(&self) -> &'static [&'static str] {
    self.restrictions
  }
}
