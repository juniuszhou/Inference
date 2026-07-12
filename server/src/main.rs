use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};

mod chat_web;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RequestPayload {
    pub message: String,
    pub value: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResponsePayload {
    pub result: String,
    pub processed_value: i32,
}

// Health check endpoint
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

// Echo endpoint - receives JSON and returns processed JSON
async fn echo(Json(payload): Json<RequestPayload>) -> impl IntoResponse {
    // let json = serde_json::from_str::<RequestPayload>();
    let response = ResponsePayload {
        result: format!("Received: {}", payload.message),
        processed_value: payload.value * 2,
    };
    (StatusCode::OK, Json(response))
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(health))
        .route("/echo", post(echo))
        .route("/v1/chat/completions", post(chat_completions))
        .merge(chat_web::routes());

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server failed");
}

// ── OpenAI-compatible Chat Completion types ──

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Chat completions endpoint — echoes the last user message for now
async fn chat_completions(Json(req): Json<ChatCompletionRequest>) -> impl IntoResponse {
    let last_user = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{:x}", rand_id()),
        object: "chat.completion".into(),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: last_user,
            },
            finish_reason: "stop".into(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    };

    (StatusCode::OK, Json(response))
}

fn rand_id() -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_payload_creation() {
        let payload = RequestPayload {
            message: "test".to_string(),
            value: 42,
        };
        assert_eq!(payload.message, "test");
        assert_eq!(payload.value, 42);
    }

    #[test]
    fn test_response_payload_creation() {
        let response = ResponsePayload {
            result: "success".to_string(),
            processed_value: 84,
        };
        assert_eq!(response.result, "success");
        assert_eq!(response.processed_value, 84);
    }

    #[test]
    fn test_json_serialization() {
        let payload = RequestPayload {
            message: "hello".to_string(),
            value: 10,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("10"));
    }

    #[test]
    fn test_json_deserialization() {
        let json = r#"{"message":"test","value":25}"#;
        let payload: RequestPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.message, "test");
        assert_eq!(payload.value, 25);
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let response = health().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_echo_endpoint() {
        let payload = RequestPayload {
            message: "hello".to_string(),
            value: 21,
        };
        let response = echo(Json(payload)).await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_response_value_doubling() {
        let input = RequestPayload {
            message: "test".to_string(),
            value: 42,
        };
        // Simulate echo behavior
        let response = ResponsePayload {
            result: format!("Received: {}", input.message),
            processed_value: input.value * 2,
        };
        assert_eq!(response.processed_value, 84);
        assert_eq!(response.result, "Received: test");
    }

    // ── OpenAI Chat Completion tests ──

    #[test]
    fn test_chat_completion_request_serialization() {
        let req = ChatCompletionRequest {
            model: "gpt-3.5-turbo".into(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "You are helpful.".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "Hello!".into(),
                },
            ],
            temperature: Some(0.7),
            max_tokens: Some(100),
            stream: Some(false),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("gpt-3.5-turbo"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_completion_deserialization() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "ping"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "gpt-4");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].content, "ping");
        assert!(req.temperature.is_none());
    }

    #[tokio::test]
    async fn test_chat_completions_echoes_last_user() {
        let req = ChatCompletionRequest {
            model: "gpt-3.5-turbo".into(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "ignore".into(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: "What is Rust?".into(),
                },
            ],
            temperature: None,
            max_tokens: None,
            stream: None,
        };
        let resp = chat_completions(Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_chat_completions_response_body() {
        let req = ChatCompletionRequest {
            model: "gpt-4".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "hello world".into(),
            }],
            temperature: None,
            max_tokens: None,
            stream: None,
        };
        let resp = chat_completions(Json(req)).await;
        let (parts, body) = resp.into_response().into_parts();
        assert_eq!(parts.status, StatusCode::OK);
        let body: ChatCompletionResponse =
            serde_json::from_slice(&axum::body::to_bytes(body, usize::MAX).await.unwrap()).unwrap();
        assert_eq!(body.object, "chat.completion");
        assert_eq!(body.choices.len(), 1);
        assert_eq!(body.choices[0].message.role, "assistant");
        assert_eq!(body.choices[0].message.content, "hello world");
        assert_eq!(body.choices[0].finish_reason, "stop");
        assert!(body.id.starts_with("chatcmpl-"));
    }
}
