#![cfg(test)]

use super::mvshmem::*;

#[test]
fn test_mvshmem_key_basic() {
    let key1 = MvshmemKey::new("test_key", 1024);
    let key2 = MvshmemKey::new("test_key", 1024);
    let key3 = MvshmemKey::new("different_key", 2048);

    assert_eq!(key1.name(), "test_key");
    assert_eq!(key1.size(), 1024);
    assert_eq!(key2, key1);
    assert_ne!(key1, key3);
}

#[test]
fn test_mvshmem_key_equality() {
    let key1 = MvshmemKey::new("same_key", 512);
    let key2 = MvshmemKey::new("same_key", 512);
    let key3 = MvshmemKey::new("same_key", 1024);

    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

#[test]
fn test_mvshmem_initialization() {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
            state.get_mut().ctx = None;
            state.get_mut().initialized = false;
        }
    }

    let result = init_mvshmem(0);
    assert!(result.is_ok(), "Failed to initialize mvshmem: {:?}", result);
}

#[test]
fn test_mvshmem_reinitialization() {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
            state.get_mut().ctx = None;
            state.get_mut().initialized = false;
        }
    }

    let result1 = init_mvshmem(0);
    assert!(result1.is_ok(), "First initialization failed: {:?}", result1);

    let result2 = init_mvshmem(0);
    assert!(result2.is_ok(), "Second initialization failed: {:?}", result2);
}

#[test]
fn test_mvshmem_write_read_basic() -> anyhow::Result<()> {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
        }
    }

    init_mvshmem(0)?;

    let key = MvshmemKey::new("test_data", 1024);

    let test_data = vec![1i32, 2i32, 3i32, 4i32, 5i32];
    write_to_mvshmem(&key, &test_data)?;

    let read_data = read_from_mvshmem(&key)?;

    assert_eq!(test_data, read_data);
    Ok(())
}

#[test]
fn test_mvshmem_different_keys() -> anyhow::Result<()> {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
        }
    }

    init_mvshmem(0)?;

    let key1 = MvshmemKey::new("key1", 1024);
    let key2 = MvshmemKey::new("key2", 1024);

    let data1 = vec![1i32, 2i32, 3i32];
    let data2 = vec![4i32, 5i32, 6i32, 7i32];

    write_to_mvshmem(&key1, &data1)?;
    write_to_mvshmem(&key2, &data2)?;

    let read_data1 = read_from_mvshmem(&key1)?;
    let read_data2 = read_from_mvshmem(&key2)?;

    assert_eq!(data1, read_data1);
    assert_eq!(data2, read_data2);
    assert_ne!(data1, data2);
    Ok(())
}

#[test]
fn test_mvshmem_write_empty() -> anyhow::Result<()> {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
        }
    }

    init_mvshmem(0)?;

    let key = MvshmemKey::new("empty_key", 1024);
    let empty_data: Vec<i32> = vec![];

    write_to_mvshmem(&key, &empty_data)?;

    let read_data = read_from_mvshmem(&key)?;
    assert_eq!(empty_data, read_data);
    Ok(())
}

#[test]
fn test_mvshmem_large_data() -> anyhow::Result<()> {
    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
        }
    }

    init_mvshmem(0)?;

    let key = MvshmemKey::new("large_key", 4096);
    let large_data: Vec<i32> = (0..100).collect();

    write_to_mvshmem(&key, &large_data)?;

    let read_data = read_from_mvshmem(&key)?;
    assert_eq!(large_data, read_data);
    Ok(())
}

#[test]
fn test_mvshmem_multithread() -> anyhow::Result<()> {
    use std::thread;
    use std::time::Duration;

    unsafe {
        if let Some(ref mut state) = MVSHMEM_STATE {
            state.get_mut().gpu_slices.clear();
        }
    }

    init_mvshmem(0)?;

    let key1 = MvshmemKey::new("thread_key_1", 1024);
    let key2 = MvshmemKey::new("thread_key_2", 1024);

    let data1 = vec![1i32, 2i32, 3i32];
    let data2 = vec![4i32, 5i32, 6i32];

    let handle1 = thread::spawn(move || {
        write_to_mvshmem(&key1, &data1).unwrap();
        thread::sleep(Duration::from_millis(10));
        read_from_mvshmem(&key1).unwrap()
    });

    let handle2 = thread::spawn(move || {
        write_to_mvshmem(&key2, &data2).unwrap();
        thread::sleep(Duration::from_millis(20));
        read_from_mvshmem(&key2).unwrap()
    });

    let result1 = handle1.join().unwrap();
    let result2 = handle2.join().unwrap();

    assert_eq!(data1, result1);
    assert_eq!(data2, result2);
    Ok(())
}
