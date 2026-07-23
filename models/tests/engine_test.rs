use models::{InferenceEngine, serve};

#[test]
fn test_serve_generates_text() {
    let mut engine = InferenceEngine::new().unwrap();

    let result = serve(&mut engine, "What is ai and llm, how to learn it?", 50);
    assert!(result.is_ok(), "serve failed: {:?}", result.err());
    let text = result.unwrap();
    assert!(!text.is_empty(), "generated text should not be empty");
    println!("Generated: {text}");
}
