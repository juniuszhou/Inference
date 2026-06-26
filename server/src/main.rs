use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

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
        .route("/echo", post(echo));

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server failed");
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
}
