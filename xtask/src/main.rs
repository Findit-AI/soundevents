use std::path::PathBuf;

use heck::{
  ToKebabCase, ToLowerCamelCase, ToShoutyKebabCase, ToShoutySnakeCase, ToShoutySnekCase,
  ToSnakeCase, ToTitleCase as _, ToTrainCase as _, ToUpperCamelCase as _,
};
use indexmap::{IndexMap, IndexSet};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use syn::parse::{Parse, Parser};

/// A tag for the audioset.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct RawSoundEvent {
  id: String,
  name: String,
  description: String,
  citation_uri: String,
  positive_examples: Vec<String>,
  child_ids: Vec<String>,
  restrictions: Vec<String>,
}

fn main() {
  let mut args = std::env::args().skip(1);
  let cmd = args.next();
  match cmd.as_deref() {
    Some("codegen") | None => codegen(),
    Some(other) => {
      eprintln!("unknown xtask command: {other}");
      eprintln!("usage: cargo xtask [codegen]");
      std::process::exit(1);
    }
  }
}

fn workspace_root() -> PathBuf {
  // CARGO_MANIFEST_DIR points to the xtask crate dir; the workspace is its parent.
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("xtask crate must live in workspace root")
    .to_path_buf()
}

fn codegen() {
  let root = workspace_root();
  let ontology_path = root.join("soundevents-dataset/assets/ontology.json");
  let output_path = root.join("soundevents-dataset/src/generated.rs");

  let data = std::fs::read_to_string(&ontology_path)
    .unwrap_or_else(|e| panic!("failed to read {}: {e}", ontology_path.display()));
  let tags: Vec<RawSoundEvent> =
    serde_json::from_str(&data).expect("failed to parse ontology.json");

  let mut consts = Vec::new();
  let mut alias_to_consts: IndexMap<String, IndexSet<syn::Ident>> = IndexMap::new();
  let mut from_code_maps = Vec::new();

  for tag in tags.iter() {
    let const_name_ident = id_to_const_name_ident(&tag.id);
    let id = tag.id.as_str();
    let name = tag.name.as_str().trim();
    let code = SipHasher::new().hash(name.as_bytes());

    alias_to_consts
      .entry(id.to_string())
      .or_default()
      .insert(const_name_ident.clone());
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
      })
      .collect::<IndexSet<_>>()
      .into_iter()
      .inspect(|s| {
        alias_to_consts
          .entry(s.clone())
          .or_default()
          .insert(const_name_ident.clone());
      })
      .collect::<Vec<_>>();

    let desp = tag.description.as_str();
    let citation_uri = if tag.citation_uri.is_empty() {
      quote! { ::core::option::Option::None }
    } else {
      let url = tag.citation_uri.as_str();
      quote! { ::core::option::Option::Some(#url) }
    };

    let restrictions = tag.restrictions.iter().map(|s| match s.as_str().trim() {
      "abstract" | "ABSTRACT" | "Abstract" => quote! { crate::Restriction::Abstract },
      "blacklist" | "BLACKLIST" | "BlackList" | "blackList" | "Blacklist" => quote! { crate::Restriction::Blacklist },
      other => panic!(
        "unknown restriction `{other}` on entry `{}`; add a new variant to `Restriction` and update xtask",
        tag.id
      ),
    });
    let children = tag.child_ids.iter().map(|id| id_to_const_name_ident(id));

    consts.push(quote! {
      const #const_name_ident: &crate::SoundEvent = &crate::SoundEvent {
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

  let phf_entries = alias_to_consts.iter().map(|(key, idents)| {
    let idents = idents.iter();
    quote! {
      #key => &[#(#idents),*]
    }
  });

  let output = quote! {
    #(#consts)*

    /// The dataset of all tags in the [https://github.com/audioset/ontology/blob/master/ontology.json](https://github.com/audioset/ontology/blob/master/ontology.json), indexed by their id, name, and aliases.
    ///
    /// Each key maps to a slice of all entries that share that name or alias.
    /// Most keys map to a single entry; a few ambiguous aliases (e.g. `"Inside"`)
    /// map to multiple entries.
    pub static DATASET: ::phf::Map<&'static ::core::primitive::str, &'static [&'static crate::SoundEvent]> = ::phf::phf_map! {
      #(#phf_entries),*
    };

    const _:() = {
      use crate::{SoundEvent, UnknownSoundEventCode};

      impl ::core::convert::TryFrom<u64> for &'static SoundEvent {
        type Error = UnknownSoundEventCode;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          SoundEvent::from_code(value).ok_or(UnknownSoundEventCode(value))
        }
      }

      impl ::core::convert::TryFrom<u64> for SoundEvent {
        type Error = UnknownSoundEventCode;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          <&'static SoundEvent>::try_from(value).cloned()
        }
      }

      impl SoundEvent {
        /// Get a tag by its code, if it exists
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub const fn from_code(id: ::core::primitive::u64) -> ::core::option::Option<&'static Self> {
          ::core::option::Option::Some(match id {
            #(#from_code_maps),*,
            _ => return ::core::option::Option::None,
          })
        }

        /// Get all entries matching an id, name, or alias.
        ///
        /// Returns an empty slice if no entries match. Most names map to a
        /// single entry, but ambiguous aliases (e.g. `"Inside"`) may return
        /// multiple entries.
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub fn from_key(name: &str) -> &'static [&'static Self] {
          match DATASET.get(name) {
            ::core::option::Option::Some(slice) => slice,
            ::core::option::Option::None => &[],
          }
        }
      }
    };
  };

  let file = prettyplease::unparse(&syn::File::parse.parse2(output).unwrap());

  let output = format!(
    r#"

// This file is generated by `cargo xtask codegen`, do not edit it manually.

{file}
"#
  );

  std::fs::write(&output_path, output)
    .unwrap_or_else(|e| panic!("failed to write {}: {e}", output_path.display()));

  println!("wrote {}", output_path.display());
}

#[inline]
fn id_to_const_name(id: &str) -> String {
  id.replace('/', "_").to_uppercase()
}

#[inline]
fn id_to_const_name_ident(id: &str) -> syn::Ident {
  format_ident!("{}", id_to_const_name(id))
}
