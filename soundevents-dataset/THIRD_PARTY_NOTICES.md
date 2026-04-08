# Third-Party Notices for `soundevents-dataset`

This crate redistributes AudioSet metadata from Google and generates static
Rust lookup tables from those upstream data files.

## AudioSet ontology

The file `assets/ontology.json` is sourced from the official AudioSet ontology
repository and is used to generate `src/ontology/generated.rs`.

- Upstream repository: <https://github.com/audioset/ontology>
- Upstream data file:
  <https://github.com/audioset/ontology/blob/master/ontology.json>
- The upstream repository states:
  "The ontology is made available by Google Inc. under a Creative Commons
  Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license."

## AudioSet released label metadata

The file `assets/class_labels_indices.csv` is sourced from the AudioSet release
metadata and is used to generate `src/rated/generated.rs`.

- AudioSet download page: <https://research.google.com/audioset/download.html>
- The AudioSet download page states:
  "The dataset is made available by Google Inc. under a Creative Commons
  Attribution 4.0 International (CC BY 4.0) license, while the ontology is
  available under a Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0) license."

These upstream files are redistributed for interoperability with AudioSet-based
models. When further redistributing this crate or derived artifacts, ensure the
applicable upstream attribution and license terms continue to be satisfied.
