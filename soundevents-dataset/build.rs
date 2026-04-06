use std::env::{self, var};

use heck::{
  ToKebabCase, ToLowerCamelCase, ToShoutyKebabCase, ToShoutySnakeCase, ToShoutySnekCase,
  ToSnakeCase, ToTitleCase as _, ToTrainCase as _, ToUpperCamelCase as _,
};
use indexmap::IndexSet;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use syn::parse::{Parse, Parser};

/// A tag for the audioset
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RawSoundEntry {
  id: String,
  name: String,
  description: String,
  citation_uri: String,
  positive_examples: Vec<String>,
  child_ids: Vec<String>,
  restrictions: Vec<String>,
}

fn main() {
  // Don't rerun this on changes other than build.rs, as we only depend on
  // the rustc version.
  println!("cargo:rerun-if-changed=build.rs");
  println!("cargo:rerun-if-changed=ontology.json");

  // Check for `--features=tarpaulin`.
  let tarpaulin = var("CARGO_FEATURE_TARPAULIN").is_ok();

  if tarpaulin {
    use_feature("tarpaulin");
  } else {
    // Always rerun if these env vars change.
    println!("cargo:rerun-if-env-changed=CARGO_TARPAULIN");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARPAULIN");

    // Detect tarpaulin by environment variable
    if env::var("CARGO_TARPAULIN").is_ok() || env::var("CARGO_CFG_TARPAULIN").is_ok() {
      use_feature("tarpaulin");
    }
  }

  // Rerun this script if any of our features or configuration flags change,
  // or if the toolchain we used for feature detection changes.
  println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TARPAULIN");

  codegen();
}

fn use_feature(feature: &str) {
  println!("cargo:rustc-cfg={}", feature);
}

#[inline]
fn codegen() {
  let data = std::fs::read_to_string("ontology.json").expect("Failed to read ontology.json");
  let tags: Vec<RawSoundEntry> =
    serde_json::from_str(&data).expect("Failed to parse ontology.json");

  let mut consts = Vec::new();
  let mut phf_maps = Vec::new();
  let mut from_code_maps = Vec::new();

  for tag in tags.iter() {
    let const_name_ident = id_to_const_name_ident(&tag.id);
    let id = tag.id.as_str();
    let name = tag.name.as_str().trim();
    let code = SipHasher::new().hash(name.as_bytes());

    phf_maps.push(quote! {
      #id => #const_name_ident
    });
    from_code_maps.push(quote! {
      #code => #const_name_ident
    });

    let aliases = tag
      .name
      .split(",")
      .flat_map(|s| {
        let default = s.trim();
        let lowercase = default.to_lowercase();
        let uppwercase = default.to_uppercase();
        let snakecase = default.to_snake_case();
        let kebab_case = default.to_kebab_case();
        let shouty_snakecase = default.to_shouty_snake_case();
        let shouty_kebab_case = default.to_shouty_kebab_case();
        let shouty_case = default.TO_SHOUTY_SNEK_CASE();
        let lower_camelase = default.to_lower_camel_case();
        let upper_camelcase = default.to_upper_camel_case();
        let title_case = default.to_title_case();
        let train_case = default.to_train_case();

        [
          name.to_string(),
          default.to_string(),
          lowercase,
          uppwercase,
          snakecase,
          kebab_case,
          shouty_snakecase,
          shouty_kebab_case,
          shouty_case,
          lower_camelase,
          upper_camelcase,
          title_case,
          train_case,
        ]
        .into_iter()
        .collect::<IndexSet<_>>()
        .into_iter()
        .map(|s| {
          phf_maps.push(quote! {
            #s => #const_name_ident
          });
          s
        })
        .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    let desp = tag.description.as_str();
    let citation_uri = if tag.citation_uri.is_empty() {
      quote! { ::core::option::Option::None }
    } else {
      let url = tag.citation_uri.as_str();
      quote! { ::core::option::Option::Some(#url) }
    };

    let restrictions = tag.restrictions.iter().map(|s| s.as_str());
    let children = tag.child_ids.iter().map(|id| id_to_const_name_ident(id));

    consts.push(quote! {
      const #const_name_ident: &crate::SoundEntry = &crate::SoundEntry {
        code: #code,
        id: #id,
        name: #name,
        aliases: &[#(#aliases),*],
        description: #desp,
        citation_uri: #citation_uri,
        children: &[#(#children),*],
        restrictions: &[#(#restrictions),*],
      };
    });
  }

  let output = quote! {
    #(#consts)*

    /// The dataset of all tags in the [https://github.com/audioset/ontology/blob/master/ontology.json](https://github.com/audioset/ontology/blob/master/ontology.json), indexed by their id, name, and aliases.
    pub static DATASET: ::phf::Map<&'static ::core::primitive::str, &'static crate::SoundEntry> = ::phf::phf_map! {
      #(#phf_maps),*
    };

    const _:() = {
      use crate::{SoundEntry, UnknownEntry, UnknownEntryCode};

      impl<'a> ::core::convert::TryFrom<&'a::core::primitive::str> for &'static SoundEntry {
        type Error = UnknownEntry<'a>;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: &'a ::core::primitive::str) -> ::core::result::Result<Self, Self::Error> {
          DATASET.get(value).copied().ok_or(UnknownEntry(value))
        }
      }

      impl<'a> ::core::convert::TryFrom<&'a::core::primitive::str> for SoundEntry {
        type Error = UnknownEntry<'a>;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: &'a ::core::primitive::str) -> ::core::result::Result<Self, Self::Error> {
          <&'static SoundEntry>::try_from(value).map(|entry| *entry)
        }
      }

      impl ::core::convert::TryFrom<u64> for &'static SoundEntry {
        type Error = UnknownEntryCode;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          SoundEntry::from_code(value).ok_or(UnknownEntryCode(value))
        }
      }

      impl ::core::convert::TryFrom<u64> for SoundEntry {
        type Error = UnknownEntryCode;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          <&'static SoundEntry>::try_from(value).map(|entry| *entry)
        }
      }

      impl SoundEntry {
        /// Get a tag by its code, if it exists
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub const fn from_code(id: ::core::primitive::u64) -> ::core::option::Option<&'static Self> {
          ::core::option::Option::Some(match id {
            #(#from_code_maps),*,
            _ => return ::core::option::Option::None,
          })
        }

        /// Get a tag by its name or alias, if it exists
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub fn from_name(name: &str) -> ::core::option::Option<&'static Self> {
          DATASET.get(name).copied()
        }
      }
    };
  };
  let file = prettyplease::unparse(&syn::File::parse.parse2(output).unwrap());

  let output = format!(
    r#"

// This file is generated by build.rs, do not edit it manually.

{file}  
"#
  );

  std::fs::write("src/generated.rs", output).expect("failed to write generated.rs");
}

#[inline]
fn id_to_const_name(id: &str) -> String {
  id.replace('/', "_").to_uppercase()
}

#[inline]
fn id_to_const_name_ident(id: &str) -> syn::Ident {
  format_ident!("{}", id_to_const_name(id))
}
