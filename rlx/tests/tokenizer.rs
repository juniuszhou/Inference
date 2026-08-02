use rlx_text::{decode_ids, load_tokenizer};

// Fixed ids for the special + vocab tokens in the toy tokenizer below.
const PAD: u32 = 0;
const BOS: u32 = 1; // <s>
const EOS: u32 = 2; // </s>
const UNK: u32 = 3;
const HELLO: u32 = 4;
const WORLD: u32 = 5;
const PAD_LEN: usize = 8;

/// A minimal HF-format `tokenizer.json`: word-level vocab, whitespace
/// pre-tokenization, `<s> … </s>` wrapping via TemplateProcessing, and
/// right-padding with `<pad>` to a fixed length of 8.
fn toy_tokenizer_json() -> String {
    r#"{
      "version": "1.0",
      "truncation": null,
      "padding": {
        "strategy": { "Fixed": 8 },
        "direction": "Right",
        "pad_to_multiple_of": null,
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "<pad>"
      },
      "added_tokens": [
        { "id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
        { "id": 1, "content": "<s>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
        { "id": 2, "content": "</s>",  "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
        { "id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
      ],
      "normalizer": null,
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": {
        "type": "TemplateProcessing",
        "single": [
          { "SpecialToken": { "id": "<s>", "type_id": 0 } },
          { "Sequence": { "id": "A", "type_id": 0 } },
          { "SpecialToken": { "id": "</s>", "type_id": 0 } }
        ],
        "pair": [
          { "SpecialToken": { "id": "<s>", "type_id": 0 } },
          { "Sequence": { "id": "A", "type_id": 0 } },
          { "SpecialToken": { "id": "</s>", "type_id": 0 } },
          { "Sequence": { "id": "B", "type_id": 1 } },
          { "SpecialToken": { "id": "</s>", "type_id": 1 } }
        ],
        "special_tokens": {
          "<s>":  { "id": "<s>",  "ids": [1], "tokens": ["<s>"] },
          "</s>": { "id": "</s>", "ids": [2], "tokens": ["</s>"] }
        }
      },
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {
          "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
          "hello": 4, "world": 5, "rlx": 6, "compiles": 7, "graphs": 8
        },
        "unk_token": "<unk>"
      }
    }"#
    .to_string()
}

#[test]
fn test_tokenizer() {
    // Materialize the tokenizer.json and load it the same way a real
    // HF checkpoint's tokenizer would be loaded.
    let path = std::env::temp_dir().join(format!(
        "rlx-usage-toy-tokenizer-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, toy_tokenizer_json()).unwrap();
    let tok = load_tokenizer(&path).unwrap();
    std::fs::remove_file(&path).ok();

    // ── Basic encode (no special tokens): words map straight to ids,
    //    then the fixed-length padding fills the tail with <pad>.
    let ids = tok.encode("hello world", false).unwrap();
    assert_eq!(ids, [HELLO, WORLD, PAD, PAD, PAD, PAD, PAD, PAD]);

    // ── With special tokens: BOS first, EOS right after the last real
    //    token, padding after that.
    let ids = tok.encode("hello world", true).unwrap();
    assert_eq!(ids.len(), PAD_LEN, "padded to fixed length");
    assert_eq!(ids[0], BOS, "starts with <s>");
    assert_eq!(&ids[1..3], [HELLO, WORLD]);
    assert_eq!(ids[3], EOS, "ends with </s> before padding");
    assert!(ids[4..].iter().all(|&i| i == PAD), "tail is all <pad>");

    // ── Attention mask marks real tokens 1 and padding 0.
    let enc = tok.raw().encode("hello world", true).unwrap();
    assert_eq!(enc.get_attention_mask(), [1, 1, 1, 1, 0, 0, 0, 0]);

    // ── Out-of-vocabulary words fall back to <unk>.
    let ids = tok.encode("hello quantum", true).unwrap();
    assert_eq!(ids[2], UNK, "'quantum' is not in the vocab");

    // ── Decode round-trip: skipping special tokens recovers the text.
    let ids = tok.encode("rlx compiles graphs", true).unwrap();
    assert_eq!(tok.decode(&ids, true).unwrap(), "rlx compiles graphs");

    // ── Decode keeping special tokens shows the full framed sequence.
    let full = decode_ids(&tok, &ids, false).unwrap();
    assert!(full.starts_with("<s>"), "full decode keeps <s>: {full}");
    assert!(full.contains("</s>"), "full decode keeps </s>: {full}");
    assert!(full.contains("<pad>"), "full decode keeps <pad>: {full}");

    println!("ids     = {ids:?}");
    println!("decoded = {full}");
}
