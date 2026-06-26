# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Build the project:**
```bash
cargo build
```

**Run the server:**
```bash
cargo run
```
The server will start on http://127.0.0.1:3000 with endpoints:
- GET /health - returns {"status": "ok"}
- POST /echo - accepts JSON {message: string, value: i32} and returns {result: string, processed_value: i32*2}

**Run tests:**
```bash
cargo test
```

**Run a specific test:**
```bash
cargo test test_name
```
Example: `cargo test test_health_endpoint`

**Run tests in release mode (for performance measurement):**
```bash
cargo test --release
```

**Check code formatting:**
```bash
cargo fmt -- --check
```

**Fix formatting:**
```bash
cargo fmt
```

**Check for lint warnings:**
```bash
cargo clippy -- -D warnings
```

## Project Structure

This is a Rust workspace with a single Axum web server:

```
Inference/
├── Cargo.toml          # Workspace configuration (members = ["server"])
├── Cargo.lock
├── server/
│   ├── Cargo.toml      # Package definition (depends on axum, tokio, serde)
│   └── src/
│       └── main.rs     # Server implementation and tests
└── target/             # Build artifacts (gitignored)
```

### server/src/main.rs

The server implements:
- **Health check endpoint** (`GET /health`): Returns JSON `{"status": "ok"}`
- **Echo endpoint** (`POST /echo`): 
  - Accepts JSON: `{"message": string, "value": i32}`
  - Returns JSON: `{"result": "Received: {message}", "processed_value": value * 2}`
- **Unit tests**: 
  - Struct serialization/deserialization tests
  - JSON serialization/deserialization tests
  - HTTP endpoint tests using Tokio runtime
  - Logic test verifying the value doubling behavior

### Dependencies (managed via workspace)
- **axum**: Web framework for building the API
- **tokio**: Async runtime (with full feature set)
- **serde**: Serialization framework (with derive feature)
- **serde_json**: JSON serialization/deserialization

## Development Notes

1. The server uses Tokio's async runtime (`#[tokio::main]`)
2. JSON handling is done via Serde's derive macros for automatic serialization/deserialization
3. All tests are contained in the same file as the implementation (standard for small Rust projects)
4. The server binds to 127.0.0.1:3000 by default
5. No external configuration is needed - all settings are hardcoded for simplicity

## Common Tasks

**Adding a new endpoint:**
1. Add a new async function that takes appropriate extractors (like Json<T>, Query<T>, or Path<T>)
2. Add a route to the Router in the main function using .route()
3. Implement the function logic
4. Add tests for the new endpoint

**Running a single test with more verbosity:**
```bash
cargo test test_name -- --nocapture
```