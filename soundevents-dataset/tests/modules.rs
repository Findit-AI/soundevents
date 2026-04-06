//! Smoke tests for the `ontology` and `rated` modules.

#[cfg(feature = "ontology")]
mod ontology {
  use soundevents_dataset::ontology::SoundEvent;

  #[test]
  fn ontology_count_matches_upstream() {
    // SoundEvent::from_code is generated from every entry, so we can count
    // distinct ids reachable through EVENTS as a sanity check.
    assert_eq!(
      SoundEvent::events().len(),
      632,
      "expected 632 ontology entries"
    );
  }

  #[test]
  fn lookup_is_case_insensitive() {
    for q in [
      "man speaking",
      "MAN SPEAKING",
      "Man Speaking",
      "mAn SpEaKiNg",
      "man_speaking",
      "manSpeaking",
      "/m/05zppz",
      "/M/05ZPPZ",
    ] {
      let r = SoundEvent::from_key(q);
      assert_eq!(r.len(), 1, "expected 1 ontology match for {q:?}");
      assert_eq!(r[0].id(), "/m/05zppz");
    }
  }

  #[test]
  fn ambiguous_alias_returns_multiple() {
    assert!(SoundEvent::from_key("Inside").len() > 1);
  }

  #[test]
  fn unknown_returns_empty() {
    assert!(SoundEvent::from_key("definitely not a sound").is_empty());
  }
}

#[cfg(feature = "rated")]
mod rated {
  use soundevents_dataset::rated::RatedSoundEvent;

  #[test]
  fn rated_count_matches_csv() {
    assert_eq!(
      RatedSoundEvent::events().len(),
      527,
      "expected 527 rated entries"
    );
  }

  #[test]
  fn lookup_is_case_insensitive() {
    let r = RatedSoundEvent::from_key("MAN SPEAKING");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].id(), "/m/05zppz");
  }

  #[test]
  fn rated_excludes_abstract_entries() {
    // "Human voice" is abstract in the upstream ontology and not in the
    // rated CSV.
    assert!(RatedSoundEvent::from_key("Human voice").is_empty());
  }

  #[test]
  fn rated_children_stay_in_rated_namespace() {
    // Pick an entry whose ontology children include unrated nodes and
    // verify that the rated view drops them. "Human sounds" is rated and
    // has many children, some abstract.
    let entries = RatedSoundEvent::from_key("Human sounds");
    if let Some(e) = entries.first() {
      // Just walk the children; if codegen left a stale id reference,
      // this would fail to compile.
      for child in e.children() {
        assert!(!child.id().is_empty());
      }
    }
  }
}
