const TEST_STACK_SIZE: usize = 8_000_000;

fn with_test_stack<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(f)
        .expect("Failed to spawn test thread")
        .join()
        .expect("Test thread panicked")
}

mod test_add;
mod test_add_2d;
mod test_argmax;
mod test_embedding;
mod test_kernels;
mod test_reduce;
