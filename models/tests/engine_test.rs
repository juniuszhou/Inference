use models::{InferenceEngine, ProcessRequest};
use tokio::sync::mpsc;

#[test]
fn test_serve_generates_text() {
    let (_tx, rx) = mpsc::channel::<ProcessRequest>(1024);
    let mut engine = InferenceEngine::new(299792458, Some(0.7), Some(0.9), rx);
    let result = engine.serve("What is ai and llm, how to learn it?", 50);
    assert!(result.is_ok(), "serve failed: {:?}", result.err());
    let text = result.unwrap();
    assert!(!text.is_empty(), "generated text should not be empty");
    println!("Generated: {text}");
}
