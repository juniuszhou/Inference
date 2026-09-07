use crate::with_test_stack;
use cutile::half::f16;
use cutile::prelude::*;
use std::sync::Arc;
use tile::kernels;

/// Build a [vocab, d] f16 table where row v is `v * 100 + col` (exact in f16).
fn make_table(vocab: usize, d: usize) -> (Arc<Tensor<f16>>, Vec<f32>) {
    let mut host = vec![f16::ZERO; vocab * d];
    let mut expect_f32 = vec![0.0f32; vocab * d];
    for v in 0..vocab {
        for c in 0..d {
            let val = (v * 100 + c) as f32;
            host[v * d + c] = f16::from_f32(val);
            expect_f32[v * d + c] = val;
        }
    }
    let t: Arc<Tensor<f16>> = api::copy_host_vec_to_device(&Arc::new(host))
        .sync()
        .expect("table h2d")
        .reshape(&[vocab, d])
        .expect("reshape table")
        .into();
    (t, expect_f32)
}

fn make_token_ids(ids: &[u32]) -> Arc<Tensor<u32>> {
    api::copy_host_vec_to_device(&Arc::new(ids.to_vec()))
        .sync()
        .expect("ids h2d")
        .into()
}

fn assert_rows_eq(host: &[f16], table_f32: &[f32], ids: &[u32], seq_len: usize, d: usize) {
    assert_eq!(host.len(), seq_len * d);
    for (r, &tok) in ids.iter().enumerate() {
        for c in 0..d {
            let got = host[r * d + c].to_f32();
            let want = table_f32[tok as usize * d + c];
            assert_eq!(got, want, "row {} col {} (token {})", r, c, tok);
        }
    }
}

#[test]
fn test_embedding_f16_single_token() {
    with_test_stack(|| {
        let (vocab, d, block) = (8usize, 32usize, 16usize);
        let ids = vec![3u32];
        let (table, table_f32) = make_table(vocab, d);
        let token_ids = make_token_ids(&ids);
        let out = api::zeros::<f16>(&[ids.len(), d]).partition([1, d]);

        let (_, _, out) = kernels::embedding_f16(token_ids, table, out)
            .generics(vec![d.to_string(), block.to_string()])
            .sync()
            .expect("embedding_f16");

        let host = out.unpartition().to_host_vec().sync().expect("out to_host");
        assert_rows_eq(&host, &table_f32, &ids, ids.len(), d);
    });
}

#[test]
fn test_embedding_f16_multi_token() {
    with_test_stack(|| {
        let (vocab, d, block) = (8usize, 32usize, 16usize);
        let ids = vec![0u32, 5, 2, 7];
        let (table, table_f32) = make_table(vocab, d);
        let token_ids = make_token_ids(&ids);
        let out = api::zeros::<f16>(&[ids.len(), d]).partition([1, d]);

        let (_, _, out) = kernels::embedding_f16(token_ids, table, out)
            .generics(vec![d.to_string(), block.to_string()])
            .sync()
            .expect("embedding_f16");

        let host = out.unpartition().to_host_vec().sync().expect("out to_host");
        assert_rows_eq(&host, &table_f32, &ids, ids.len(), d);
    });
}

#[test]
fn test_embedding_f16_3_batched() {
    with_test_stack(|| {
        let (vocab, n, d, block) = (8usize, 2usize, 32usize, 16usize);
        let ids = vec![1u32, 4, 0, 7]; // seq_len=4, N=2 -> 2 CTAs
        let (table, table_f32) = make_table(vocab, d);
        let token_ids = make_token_ids(&ids);
        let out = api::zeros::<f16>(&[ids.len(), d]).partition([n, d]);

        let (_, _, out) = kernels::embedding_f16_3(token_ids, table, out)
            .generics(vec![n.to_string(), d.to_string(), block.to_string()])
            .sync()
            .expect("embedding_f16_3");

        let host = out.unpartition().to_host_vec().sync().expect("out to_host");
        assert_rows_eq(&host, &table_f32, &ids, ids.len(), d);
    });
}

#[test]
fn test_embedding_f16_3_matches_f16() {
    with_test_stack(|| {
        let (vocab, n, d, block) = (8usize, 2usize, 32usize, 16usize);
        let ids = vec![6u32, 3, 5, 1];
        let (table, _) = make_table(vocab, d);

        let out1 = api::zeros::<f16>(&[ids.len(), d]).partition([1, d]);
        let (_, _, out1) = kernels::embedding_f16(make_token_ids(&ids), table.dup(), out1)
            .generics(vec![d.to_string(), block.to_string()])
            .sync()
            .expect("embedding_f16");
        let host1 = out1
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("out1 to_host");

        let out3 = api::zeros::<f16>(&[ids.len(), d]).partition([n, d]);
        let (_, _, out3) = kernels::embedding_f16_3(make_token_ids(&ids), table, out3)
            .generics(vec![n.to_string(), d.to_string(), block.to_string()])
            .sync()
            .expect("embedding_f16_3");
        let host3 = out3
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("out3 to_host");

        assert_eq!(host1.len(), host3.len());
        for (i, (a, b)) in host1.iter().zip(host3.iter()).enumerate() {
            assert_eq!(a.to_f32(), b.to_f32(), "element {}", i);
        }
    });
}
