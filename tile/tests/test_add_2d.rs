use cutile::prelude::*;
use std::sync::Arc;
use tile::kernels;

#[test]
fn test_add_2d_ones() {
    // BM=4, BN=2, BK=8, M=8
    // x and y are [M, BK] = [8, 8] all ones.
    // Kernel loops BK/BN = 4 times, each `z.store` overwrites the same
    // [BM, BN] output tile, so only the last iteration survives:
    // each element = 1.0 + 1.0 = 2.0.
    let (bm, bn, bk) = (4usize, 2usize, 8usize);
    let m = 8usize;

    let x = api::ones::<f32>(&[m, bk]);
    let y = api::ones::<f32>(&[m, bk]);
    let z = api::zeros::<f32>(&[bm, bn]).partition([bm, bn]);

    let (z, _, _) = kernels::add_2d(z, x, y)
        .generics(vec![bm.to_string(), bn.to_string(), bk.to_string()])
        .sync()
        .expect("add_2d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![2.0; bm * bn]);
}

#[test]
fn test_add_2d_sequential() {
    // BM=2, BN=2, BK=4, M=4 single output tile.
    // x = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] row-major [4,4]
    // Partition [BM, BN]=[2,2] yields tiles:
    //   [0,0]=[1,2, 5,6]  [0,1]=[3,4, 7,8]
    //   [1,0]=[9,10,13,14] [1,1]=[11,12,15,16]
    // With a single-tile z ([2,2]), only pid.0=0 runs. It loads [0,0] then
    // [0,1] and overwrites, so result is [0,1]+1 = [4,5,8,9].
    let (bm, bn, bk) = (2usize, 2usize, 4usize);
    let m = 4usize;

    let mut x_data = Vec::new();
    for i in 0..m {
        for j in 0..bk {
            x_data.push((i * bk + j) as f32 + 1.0);
        }
    }
    let x = api::copy_host_vec_to_device(&Arc::new(x_data))
        .sync()
        .expect("x h2d")
        .reshape(&[m, bk])
        .expect("reshape x");

    let y = api::ones::<f32>(&[m, bk]);
    let z = api::zeros::<f32>(&[bm, bn]).partition([bm, bn]);

    let (z, _, _) = kernels::add_2d(z, x, y)
        .generics(vec![bm.to_string(), bn.to_string(), bk.to_string()])
        .sync()
        .expect("add_2d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");

    let expected = vec![4.0, 5.0, 8.0, 9.0];
    assert_eq!(z_host, expected);
}

#[test]
fn test_add_2d_sequential_multi_block() {
    // Same data as above but with a multi-block output to verify pid.0
    // dispatch: z is [M, BN]=[4,2] partitioned [2,2] -> 2 blocks (rows).
    // pid.0=0 loads [0,0],[0,1] -> last tile [3,4,7,8]+1 = [4,5,8,9]
    // pid.0=1 loads [1,0],[1,1] -> last tile [11,12,15,16]+1 = [12,13,16,17]
    let (bm, bn, bk) = (2usize, 2usize, 4usize);
    let m = 4usize;

    let mut x_data = Vec::new();
    for i in 0..m {
        for j in 0..bk {
            x_data.push((i * bk + j) as f32 + 1.0);
        }
    }
    let x = api::copy_host_vec_to_device(&Arc::new(x_data))
        .sync()
        .expect("x h2d")
        .reshape(&[m, bk])
        .expect("reshape x");

    let y = api::ones::<f32>(&[m, bk]);
    // multi-block output: [M, BN] = [4, 2]
    let z = api::zeros::<f32>(&[m, bn]).partition([bm, bn]);

    let (z, _, _) = kernels::add_2d(z, x, y)
        .generics(vec![bm.to_string(), bn.to_string(), bk.to_string()])
        .sync()
        .expect("add_2d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");

    let expected = vec![
        4.0, 5.0, // row 0: tile [0,1] + 1
        8.0, 9.0, // row 1
        12.0, 13.0, // row 2: tile [1,1] + 1
        16.0, 17.0, // row 3
    ];
    assert_eq!(z_host, expected);
}

#[test]
fn test_add_2d_non_divisible_bk_bn() {
    // BK=6, BN=2 => BK/BN=3 tiles per block, M=4.
    // Single-tile z: only last column-block survives, each element 1+1=2.
    let (bm, bn, bk) = (2usize, 2usize, 6usize);
    let m = 4usize;

    let x = api::ones::<f32>(&[m, bk]);
    let y = api::ones::<f32>(&[m, bk]);
    let z = api::zeros::<f32>(&[bm, bn]).partition([bm, bn]);

    let (z, _, _) = kernels::add_2d(z, x, y)
        .generics(vec![bm.to_string(), bn.to_string(), bk.to_string()])
        .sync()
        .expect("add_2d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");
    assert_eq!(z_host, vec![2.0; bm * bn]);
}

#[test]
fn test_add_2d_single_tile() {
    // BK=BN => loop runs once, no overwrite effect.
    let (bm, bn, bk) = (4usize, 4usize, 4usize);
    let m = 4usize;

    let mut x_data = Vec::new();
    for i in 0..m {
        for j in 0..bk {
            x_data.push((i * bk + j) as f32 + 1.0);
        }
    }
    let x = api::copy_host_vec_to_device(&Arc::new(x_data))
        .sync()
        .expect("x h2d")
        .reshape(&[m, bk])
        .expect("reshape x");

    let y = api::ones::<f32>(&[m, bk]);
    let z = api::zeros::<f32>(&[bm, bn]).partition([bm, bn]);

    let (z, _, _) = kernels::add_2d(z, x, y)
        .generics(vec![bm.to_string(), bn.to_string(), bk.to_string()])
        .sync()
        .expect("add_2d");
    let z_host = z.unpartition().to_host_vec().sync().expect("z to_host");

    // BK/BN=1, so each element = x[i][j] + 1.0
    let expected: Vec<f32> = (0..bm * bn).map(|i| (i as f32 + 1.0) + 1.0).collect();
    assert_eq!(z_host, expected);
}
