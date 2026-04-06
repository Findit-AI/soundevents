use std::{collections::HashSet, path::PathBuf};

use heck::{
  ToKebabCase, ToLowerCamelCase, ToShoutyKebabCase, ToShoutySnakeCase, ToShoutySnekCase,
  ToSnakeCase, ToTitleCase as _, ToTrainCase as _, ToUpperCamelCase as _,
};
use indexmap::{IndexMap, IndexSet};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher;
use syn::parse::{Parse, Parser};
use uncased::UncasedStr;

/// A tag for the audioset (mirrors `ontology.json` schema).
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

/// One row of `class_labels_indices.csv`.
#[derive(Debug, Deserialize)]
struct CsvRow {
  #[allow(dead_code)]
  index: u32,
  mid: String,
  #[allow(dead_code)]
  display_name: String,
}

/// Per-entry data extracted from `ontology.json` once and reused to emit
/// both the `ontology` and the `rated` modules.
#[derive(Debug, Clone)]
struct EntryRecord {
  id: String,
  const_ident: syn::Ident,
  code: u64,
  name: String,
  /// Alias variants (original casing) — stored on the struct's `aliases` field.
  alias_strings: Vec<String>,
  /// Lowercased phf keys for this entry (id + alias variants, deduped).
  phf_keys: Vec<String>,
  description: String,
  citation_uri: Option<String>,
  /// Restriction enum tokens (one per restriction string).
  restrictions: Vec<TokenStream>,
  child_ids: Vec<String>,
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
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .expect("xtask crate must live in workspace root")
    .to_path_buf()
}

fn codegen() {
  let root = workspace_root();
  let ontology_json = root.join("soundevents-dataset/assets/ontology.json");
  let csv_path = root.join("soundevents-dataset/assets/class_labels_indices.csv");
  let ontology_out = root.join("soundevents-dataset/src/ontology/generated.rs");
  let rated_out = root.join("soundevents-dataset/src/rated/generated.rs");

  // 1. Parse ontology.json into per-entry records (one pass).
  let raw_data = std::fs::read_to_string(&ontology_json)
    .unwrap_or_else(|e| panic!("failed to read {}: {e}", ontology_json.display()));
  let tags: Vec<RawSoundEvent> =
    serde_json::from_str(&raw_data).expect("failed to parse ontology.json");
  let records: Vec<EntryRecord> = tags.iter().map(build_record).collect();

  // 2. Parse the rated CSV into a set of mids (preserves CSV ordering, but
  //    we only need set membership for filtering).
  let rated_ids = read_rated_ids(&csv_path);

  // 3. Emit the full ontology module (all 632 records).
  let all_ids: HashSet<&str> = records.iter().map(|r| r.id.as_str()).collect();
  emit_module(
    &records,
    &all_ids,
    "SoundEvent",
    "UnknownSoundEventCode",
    "ontology",
    &ontology_out,
  );

  // 4. Emit the rated module (only the 527 entries in the CSV; their child
  //    links are filtered to other rated entries).
  let rated_set: HashSet<&str> = rated_ids.iter().map(String::as_str).collect();
  emit_module(
    &records,
    &rated_set,
    "RatedSoundEvent",
    "UnknownRatedSoundEventCode",
    "rated",
    &rated_out,
  );
}

fn read_rated_ids(path: &PathBuf) -> Vec<String> {
  let mut rdr = csv::ReaderBuilder::new()
    .has_headers(true)
    .from_path(path)
    .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
  let mut out = Vec::new();
  for row in rdr.deserialize::<CsvRow>() {
    out.push(row.expect("failed to parse CSV row").mid);
  }
  out
}

fn build_record(tag: &RawSoundEvent) -> EntryRecord {
  let const_ident = id_to_const_name_ident(&tag.id);
  let id = tag.id.trim().to_string();
  let name = tag.name.trim().to_string();
  let code = SipHasher::new().hash(name.as_bytes());

  // Alias variants for the struct's `aliases` field — original casing,
  // deduped within the entry by exact-string equality.
  let alias_strings: Vec<String> = tag
    .name
    .split(',')
    .flat_map(|s| {
      let default = s.trim();
      [
        name.clone(),
        default.to_string(),
        default.to_lowercase(),
        default.to_uppercase(),
        default.to_snake_case(),
        default.to_kebab_case(),
        default.to_shouty_snake_case(),
        default.to_shouty_kebab_case(),
        default.TO_SHOUTY_SNEK_CASE(),
        default.to_lower_camel_case(),
        default.to_upper_camel_case(),
        default.to_title_case(),
        default.to_train_case(),
        to_sentence_case(default),
      ]
    })
    .collect::<IndexSet<_>>()
    .into_iter()
    .collect();

  // PHF keys: lowercased forms (because the map is `UncasedStr`-keyed),
  // deduped, including the entry's id.
  let mut phf_keys: Vec<String> = Vec::new();
  let mut seen = IndexSet::<String>::new();
  for s in std::iter::once(id.clone()).chain(alias_strings.iter().cloned()) {
    let lower = s.to_lowercase();
    if seen.insert(lower.clone()) {
      phf_keys.push(lower);
    }
  }

  let citation_uri = if tag.citation_uri.trim().is_empty() {
    None
  } else {
    Some(tag.citation_uri.clone())
  };

  let restrictions: Vec<TokenStream> = tag
    .restrictions
    .iter()
    .map(|s| match s.as_str().trim() {
      "abstract" | "ABSTRACT" | "Abstract" => quote! { crate::Restriction::Abstract },
      "blacklist" | "BLACKLIST" | "BlackList" | "blackList" | "Blacklist" => {
        quote! { crate::Restriction::Blacklist }
      }
      other => panic!(
        "unknown restriction `{other}` on entry `{}`; add a new variant to `Restriction` and update xtask",
        tag.id
      ),
    })
    .collect();

  EntryRecord {
    id,
    const_ident,
    code,
    name,
    alias_strings,
    phf_keys,
    description: tag.description.clone(),
    citation_uri,
    restrictions,
    child_ids: tag.child_ids.clone(),
  }
}

/// Emit one generated.rs for a module (`ontology` or `rated`).
///
/// Only entries whose id is in `included_ids` are emitted. The `children`
/// field of each emitted entry is filtered to references that are also in
/// `included_ids`, so traversing the hierarchy stays inside the module.
fn emit_module(
  records: &[EntryRecord],
  included_ids: &HashSet<&str>,
  type_name: &str,
  err_name: &str,
  module_name: &str,
  output_path: &PathBuf,
) {
  let type_ident = format_ident!("{}", type_name);
  let err_ident = format_ident!("{}", err_name);

  let mut consts = Vec::new();
  let mut from_code_arms = Vec::new();
  // alias_to_consts: lowercased phf key -> set of const idents pointing at it.
  let mut alias_to_consts: IndexMap<String, IndexSet<syn::Ident>> = IndexMap::new();

  for record in records {
    if !included_ids.contains(record.id.as_str()) {
      continue;
    }
    let const_name_ident = &record.const_ident;
    let id = &record.id;
    let name = &record.name;
    let code = record.code;
    let desp = &record.description;
    let citation_uri = match &record.citation_uri {
      Some(url) => quote! { ::core::option::Option::Some(#url) },
      None => quote! { ::core::option::Option::None },
    };
    let aliases = record.alias_strings.iter().map(|s| s.as_str());
    let restrictions = record.restrictions.iter();
    // Filter children to those still inside the included set.
    let children = record
      .child_ids
      .iter()
      .filter(|c| included_ids.contains(c.as_str()))
      .map(|c| id_to_const_name_ident(c));

    consts.push(quote! {
      const #const_name_ident: &super::#type_ident = &super::#type_ident {
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

    from_code_arms.push(quote! {
      #code => #const_name_ident
    });

    for key in &record.phf_keys {
      alias_to_consts
        .entry(key.clone())
        .or_default()
        .insert(const_name_ident.clone());
    }
  }

  // Build the perfect-hash map with phf_codegen, keyed by &UncasedStr so
  // lookups are case-insensitive.
  let mut phf_map = phf_codegen::Map::<&UncasedStr>::new();
  let value_strings: Vec<(String, String)> = alias_to_consts
    .iter()
    .map(|(key, idents)| {
      let inner = idents
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
      (key.clone(), format!("__slice(&[{inner}])"))
    })
    .collect();
  for (key, value) in &value_strings {
    phf_map.entry(UncasedStr::new(key.as_str()), value);
  }
  let phf_built = phf_map.build().to_string();

  // The handcoded chunk of generated.rs (consts + helper + impls). The phf
  // static is appended afterwards as raw text from `phf_built`.
  let body = quote! {
    #(#consts)*

    #[doc(hidden)]
    const fn __slice(
      s: &'static [&'static super::#type_ident],
    ) -> &'static [&'static super::#type_ident] {
      s
    }

    const _: () = {
      use super::{#type_ident, #err_ident};

      impl ::core::convert::TryFrom<u64> for &'static #type_ident {
        type Error = #err_ident;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          #type_ident::from_code(value).ok_or(#err_ident(value))
        }
      }

      impl ::core::convert::TryFrom<u64> for #type_ident {
        type Error = #err_ident;

        #[cfg_attr(not(tarpaulin), inline(always))]
        fn try_from(value: u64) -> ::core::result::Result<Self, Self::Error> {
          <&'static #type_ident>::try_from(value).cloned()
        }
      }

      impl #type_ident {
        /// Get an entry by its code, if it exists.
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub const fn from_code(id: ::core::primitive::u64) -> ::core::option::Option<&'static Self> {
          ::core::option::Option::Some(match id {
            #(#from_code_arms),*,
            _ => return ::core::option::Option::None,
          })
        }

        /// Get all entries matching an id, name, or alias.
        ///
        /// Lookups are case-insensitive: `"man speaking"`, `"MAN SPEAKING"`,
        /// and `"Man Speaking"` all resolve to the same entry. Separator
        /// styles (`"man_speaking"`, `"man-speaking"`, `"manSpeaking"`) are
        /// each indexed separately.
        ///
        /// Returns an empty slice if no entries match. Most names map to a
        /// single entry, but ambiguous aliases (e.g. `"Inside"`) may return
        /// multiple entries.
        #[cfg_attr(not(tarpaulin), inline(always))]
        pub fn from_key(name: &str) -> &'static [&'static Self] {
          match DATASET.get(::uncased::UncasedStr::new(name)) {
            ::core::option::Option::Some(slice) => slice,
            ::core::option::Option::None => &[],
          }
        }
      }
    };
  };

  let body_pretty = prettyplease::unparse(&syn::File::parse.parse2(body).unwrap());

  let phf_static = format!(
    "use ::uncased::UncasedStr;\n\n\
     /// All {module_name} entries, indexed by id, name, and alias.\n\
     ///\n\
     /// Lookups are case-insensitive (the keys are [`UncasedStr`]), so any\n\
     /// case form of an alias resolves through the same bucket. Each key\n\
     /// maps to a slice of all entries that share that name or alias —\n\
     /// most keys map to a single entry, but a few ambiguous aliases\n\
     /// (e.g. `\"Inside\"`) may map to multiple entries.\n\
     pub(super) static DATASET: ::phf::Map<&'static UncasedStr, &'static [&'static super::{type_name}]> = {phf_built};\n",
  );

  let output = format!(
    "\n\n// This file is generated by `cargo xtask codegen`, do not edit it manually.\n\n{body_pretty}\n{phf_static}\n",
  );

  std::fs::write(output_path, output)
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

/// Sentence case: lowercase the whole string, then uppercase the first character.
/// For `"man speaking"` this yields `"Man speaking"`. heck has no built-in for
/// this style — `to_title_case` would produce `"Man Speaking"` instead.
fn to_sentence_case(s: &str) -> String {
  let lower = s.to_lowercase();
  let mut chars = lower.chars();
  match chars.next() {
    Some(first) => first.to_uppercase().chain(chars).collect(),
    None => String::new(),
  }
}
