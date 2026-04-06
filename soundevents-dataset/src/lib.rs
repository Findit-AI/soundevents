#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

#[cfg(feature = "ontology")]
#[cfg_attr(docsrs, doc(cfg(feature = "ontology")))]
pub mod ontology;

#[cfg(feature = "rated")]
#[cfg_attr(docsrs, doc(cfg(feature = "rated")))]
pub mod rated;

#[cfg(feature = "ontology")]
#[cfg_attr(docsrs, doc(cfg(feature = "ontology")))]
pub use ontology::SoundEvent;

#[cfg(feature = "rated")]
#[cfg_attr(docsrs, doc(cfg(feature = "rated")))]
pub use rated::RatedSoundEvent;

/// Errors that can occur when parsing a [`Restriction`] from a string.
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[error("unknown restriction: {0}")]
pub struct UnknownRestriction<'a>(&'a str);

impl<'a> UnknownRestriction<'a> {
  /// Get the name associated with the `UnknownRestriction` error
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn name(&self) -> &'a str {
    self.0
  }
}

impl<'a> TryFrom<&'a str> for Restriction {
  type Error = UnknownRestriction<'a>;

  #[cfg_attr(not(tarpaulin), inline(always))]
  fn try_from(value: &'a str) -> Result<Self, Self::Error> {
    Ok(match value {
      "abstract" | "ABSTRACT" | "Abstract" => Restriction::Abstract,
      "blacklist" | "BLACKLIST" | "BlackList" | "blackList" | "Blacklist" => Restriction::Blacklist,
      _ => return Err(UnknownRestriction(value)),
    })
  }
}

/// A restriction on a sound entry, which may be an abstract category or a blacklisted entry
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum Restriction {
  /// For a class that is principally a container within the hierarchy, but will not have any explicit examples for itself. "Human voice" is an abstract class. Abstract classes will always have children.
  Abstract,
  /// For classes that have been excluded from rating for the time being. These are classes that we found were too difficult for raters to mark reliably, or for which we had too much trouble finding candidates, or which we decided to drop from labeling for some other reason.
  Blacklist,
}

impl core::fmt::Display for Restriction {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{}", self.as_str())
  }
}

impl Restriction {
  /// Get the string representation of the restriction
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Restriction::Abstract => "abstract",
      Restriction::Blacklist => "blacklist",
    }
  }

  /// Return `true` if the restriction is an abstract category
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_abstract(&self) -> bool {
    matches!(self, Restriction::Abstract)
  }

  /// Return `true` if the restriction is a blacklisted entry.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_blacklist(&self) -> bool {
    matches!(self, Restriction::Blacklist)
  }
}

/// Defines a sound-event struct (with all its accessors), its companion
/// `Unknown*Code` error type, and `Display`. Used twice — once in the
/// `ontology` module and once in the `rated` module — to produce two
/// independent types with identical shape.
macro_rules! define_sound_event {
  (
    $(#[$struct_meta:meta])*
    name: $name:ident,
    $(#[$err_meta:meta])*
    error: $err_name:ident,
    error_message: $err_msg:literal
    $(,
      extra_fields: {
        $($extra_field:tt)*
      }
    )?
    $(,
      extra_impl: {
        $($extra_impl:tt)*
      }
    )?
    $(,)?
  ) => {
    $(#[$struct_meta])*
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize))]
    pub struct $name {
      #[cfg_attr(feature = "serde", serde(skip))]
      pub(crate) code: u64,
      pub(crate) id: &'static str,
      pub(crate) name: &'static str,
      #[cfg_attr(feature = "serde", serde(skip_serializing_if = "<[_]>::is_empty"))]
      pub(crate) aliases: &'static [&'static str],
      pub(crate) description: &'static str,
      #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
      pub(crate) citation_uri: ::core::option::Option<&'static str>,
      #[cfg_attr(feature = "serde", serde(skip_serializing_if = "<[_]>::is_empty"))]
      pub(crate) children: &'static [&'static $name],
      #[cfg_attr(feature = "serde", serde(skip_serializing_if = "<[_]>::is_empty"))]
      pub(crate) restrictions: &'static [$crate::Restriction],
      $($($extra_field)*)?
    }

    impl ::core::fmt::Display for $name {
      #[cfg_attr(not(tarpaulin), inline(always))]
      fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.write_str(self.name)
      }
    }

    impl $name {
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
      pub const fn citation_uri(&self) -> ::core::option::Option<&'static str> {
        self.citation_uri
      }

      /// Get the sound entry's children sound entries
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn children(&self) -> &'static [&'static Self] {
        self.children
      }

      /// Get the sound entry's restrictions
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn restrictions(&self) -> &'static [$crate::Restriction] {
        self.restrictions
      }

      $($($extra_impl)*)?
    }

    $(#[$err_meta])*
    #[derive(Debug, ::thiserror::Error, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[error($err_msg)]
    pub struct $err_name(pub(crate) u64);

    impl $err_name {
      /// Get the code associated with this error.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn code(&self) -> u64 {
        self.0
      }
    }
  };
}

pub(crate) use define_sound_event;
