use soundevents_dataset::SoundEvent;

#[test]
fn case_insensitive_lookup() {
  let queries = [
    "man speaking",
    "MAN SPEAKING",
    "Man Speaking",
    "mAn SpEaKiNg",
    "man_speaking",
    "MAN_SPEAKING",
    "manSpeaking",
    "MANSPEAKING",
    "man-speaking",
    "Man-Speaking",
    "/m/05zppz",
    "/M/05ZPPZ",
  ];
  for q in queries {
    let r = SoundEvent::from_key(q);
    assert_eq!(r.len(), 1, "expected 1 match for {q:?}");
    assert_eq!(r[0].id(), "/m/05zppz");
  }
}

#[test]
fn unknown_returns_empty() {
  assert!(SoundEvent::from_key("definitely not a sound").is_empty());
}

#[test]
fn ambiguous_alias_returns_multiple() {
  assert!(SoundEvent::from_key("Inside").len() > 1);
  assert!(SoundEvent::from_key("INSIDE").len() > 1);
  assert!(SoundEvent::from_key("inside").len() > 1);
}
